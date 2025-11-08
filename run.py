# -*- coding: utf-8 -*-
"""
FEED-NIDS End-to-End Pipeline (Detection → Prioritizer → EvidencePack → RAG → Decision → EvidenceWriter)
- Round_k 순차 실행 → Evidence 저장 → RAG 재인덱싱 → 다음 라운드 반영
- ★ Label 재조정 로직 추가: 유사 사례 기반 Label 업데이트
- 라운드별 산출물: round_outputs/<Round>/*
"""
import os
import glob
import json
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any

try:
    import yaml
    _YAML_OK = True
except Exception:
    _YAML_OK = False

import pandas as pd

from tools.detection import DetectionTool
from tools.prioritizer import PrioritizerTool
from tools.evidence_pack import EvidencePackTool
from tools.feedback_search import FeedbackSearchTool
from orchestrator.policy import DecisionPolicy
from tools.evidence_writer import EvidenceWriter

def load_config(config_path: str = "./configs/config.yaml") -> Dict[str, Any]:
    default_cfg = {
        "paths": {
            "model_path": "./orchestrator/models/xgboost_binary_classifier.joblib",
            "predictions_dir": "./round_predictions",
            "feedback_cases_dir": "./feedback_cases",
            "results_dir": "./round_results",
            "evidence_db_dir": "./evidence_db",
            "evidence_index_file": "evidence_index.jsonl",
            "evidence_packs_file": "packs.jsonl",
            "round_outputs_dir": "./round_outputs",
        },
        "gating": {"bottom_percent": 5.0, "alpha": 0.3, "beta": 0.7},
        "evidence_pack": {"shap_topk": 5},
        "rag": {
            "index_type": "hnsw",
            "coarse_k": 200,
            "hnsw": {"M": 32, "efSearch": 128},
            "ivf_pq": {"nlist": 8192, "nprobe": 32, "pq_m": 64, "pq_bits": 8},
            "enable_prefilter": True,
            "reload_after_each_round": True
        },
        "decision": {
            "alpha_ip": 0.5, 
            "beta_cos": 0.3, 
            "gamma_overlap": 0.2,
            "threshold": 0.7, 
            "top_refs": 3,
            "label_confidence_threshold": 0.6  # ★ Label 재조정 임계값
        },
        "evidence": {
            "not_found_sim_threshold": 0.15, 
            "insertion_rate_percent": 3.0,  # ★ 3%로 변경
            "dedup_round_digits": 3
        },
        "pipeline": {"pause_between_rounds": False}
    }
    if os.path.exists(config_path) and _YAML_OK:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        def merge(a,b):
            for k,v in b.items():
                if k not in a:
                    a[k] = v
                elif isinstance(v, dict) and isinstance(a[k], dict):
                    merge(a[k], v)
        merge(cfg, default_cfg)
        return cfg
    return default_cfg

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in (rows or []):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _append_lines(path: str, lines):
    if not lines: return
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines: f.write(str(ln).strip() + "\n")

def _append_round_log(csv_path: str, row_dict: dict):
    header = ["round","gated","similar_found","label_adjusted","not_found","inserted","packs_total_after_rebuild"]
    _ensure_dir(os.path.dirname(csv_path))
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow({k: row_dict.get(k, "") for k in header})

def load_evidence_packs_from_jsonl(db_dir: str, packs_file: str = "packs.jsonl"):
    path = os.path.join(db_dir, packs_file)
    if not os.path.exists(path):
        print(f"[RAG] packs.jsonl 없음: {path} (빈 인덱스로 시작)")
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    print(f"[RAG] Evidence Packs 로드: {len(out):,}개")
    return out

class FeedNidsOrchestrator:
    def __init__(self, det, pri, epk, rag, policy, evidence_writer):
        self.det = det
        self.pri = pri
        self.epk = epk
        self.rag = rag
        self.policy = policy
        self.writer = evidence_writer

    async def handle_predictions_csv(self, predictions_csv: str, round_name: str, out_dir: str) -> dict:
        _ensure_dir(out_dir)
        df_pred = pd.read_csv(predictions_csv, low_memory=False)

        # 2) 게이팅
        targets_df = self.pri.select(df_pred, round_name=round_name)
        if targets_df.empty:
            print(f"[{round_name}] 게이팅 결과 대상 없음")
            targets_df.to_csv(os.path.join(out_dir, "targets.csv"), index=False)
            for fn in ["evidence_packs.jsonl","rag_candidates.jsonl","decisions.jsonl"]:
                _write_jsonl(os.path.join(out_dir, fn), [])
            open(os.path.join(out_dir, "evidence_inserted_ids.txt"), "w").close()
            json.dump({}, open(os.path.join(out_dir, "round_summary.json"), "w", encoding="utf-8"), ensure_ascii=False)
            return {"decisions": [], "inserted": 0, "gated": 0, "not_found": 0, 
                   "similar_found": 0, "label_adjusted": 0}

        targets_df.to_csv(os.path.join(out_dir, "targets.csv"), index=False)
        
        # alert_id 확인 및 생성
        if "alert_id" not in targets_df.columns:
            if "case_id" in targets_df.columns:
                targets_df["alert_id"] = targets_df["case_id"]
            else:
                targets_df["alert_id"] = [f"{round_name}_{i:06d}" for i in range(len(targets_df))]
            print(f"[{round_name}] alert_id 컬럼 생성됨")

        # 3) Evidence Pack 생성
        packs = self.epk.build_from_targets(targets_df)
        _write_jsonl(os.path.join(out_dir, "evidence_packs.jsonl"), packs)

        # 4) RAG 검색 & 결정 (★ Label 재조정 포함)
        rag_map = {}
        decisions = []
        rag_dump_rows = []
        
        similar_found_count = 0
        label_adjusted_count = 0
        
        for ep in packs:
            # RAG 검색
            res = self.rag.query(ep, k=self.rag.coarse_k)
            rag_map[ep["alert_id"]] = res
            
            # ★ 실제 Ground Truth Label 가져오기 (targets_df에서)
            alert_id = ep["alert_id"]
            matching_rows = targets_df[targets_df["alert_id"] == alert_id]
            
            if not matching_rows.empty:
                ground_truth_label = matching_rows.iloc[0].get("label", "Unknown")
                predicted_label = matching_rows.iloc[0].get("predicted_label", "Attack")
            else:
                ground_truth_label = "Unknown"
                predicted_label = "Attack"
            
            # ★ Decision Policy로 Label 재조정
            # original_label = 예측값 (게이팅 단계에서는 모두 Attack)
            # 하지만 Evidence에는 ground_truth를 저장해야 함!
            d = self.policy.decide(
                alert_id=ep["alert_id"], 
                candidates=res.get("candidates", []),
                original_label=predicted_label  # 예측값 기준으로 재조정 판단
            )
            # ★ Ground Truth를 decision에 추가
            d["ground_truth_label"] = ground_truth_label
            decisions.append(d)
            
            # 통계
            if res.get("candidates"):
                similar_found_count += 1
            if d.get("adjusted_label") != predicted_label:
                label_adjusted_count += 1
            
            rag_dump_rows.append({
                "alert_id": ep["alert_id"],
                "candidates": (res.get("candidates", [])[:20] if res else [])
            })
        
        _write_jsonl(os.path.join(out_dir, "rag_candidates.jsonl"), rag_dump_rows)
        _write_jsonl(os.path.join(out_dir, "decisions.jsonl"), decisions)
        
        # ★ Decision 타입별 통계
        decision_stats = {}
        for d in decisions:
            dt = d.get("decision", "UNKNOWN")
            decision_stats[dt] = decision_stats.get(dt, 0) + 1
        print(f"[{round_name}] Decision 분포: {decision_stats}")

        # 5) 미검색 케이스 선정 → Evidence DB 저장 (★ Label 포함)
        not_found = self.writer.select_not_found(packs, rag_map, decisions)
        
        # 통계 출력
        if not_found:
            reason_counts = {}
            for nf in not_found:
                reason = nf.get("evidence_reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            print(f"[{round_name}] Evidence 선정 사유:")
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count}개")
        
        to_insert = self.writer.subsample(not_found)
        saved_ids = self.writer.write(to_insert)
        _append_lines(os.path.join(out_dir, "evidence_inserted_ids.txt"), saved_ids)

        # 6) adjusted_label 반영 파일 생성
        if "alert_id" in df_pred.columns:
            dec_df = pd.DataFrame(decisions)
            
            # alert_id 기준으로 merge (targets에 있는 것만)
            target_ids = set(targets_df["alert_id"].tolist()) if "alert_id" in targets_df.columns else set()
            dec_df_filtered = dec_df[dec_df["alert_id"].isin(target_ids)]
            
            df_pred_out = df_pred.merge(
                dec_df_filtered[["alert_id", "adjusted_label", "confidence", "decision"]], 
                on="alert_id", 
                how="left"
            )
            df_pred_out.to_csv(os.path.join(out_dir, "predictions_with_adjusted.csv"), index=False)
            
            # Label 변경 통계 출력
            changed = dec_df_filtered[dec_df_filtered["adjusted_label"] != dec_df_filtered["original_label"]]
            if not changed.empty:
                print(f"[{round_name}] Label 재조정: {len(changed)}건")
                print(f"  - Normal로 변경: {len(changed[changed['adjusted_label']=='Normal'])}건")
                print(f"  - Attack 유지: {len(changed[changed['adjusted_label']=='Attack'])}건")

        # 요약
        summary = {
            "round": round_name, 
            "gated": int(len(packs)),
            "similar_found": similar_found_count,
            "label_adjusted": label_adjusted_count,
            "not_found": int(len(not_found)), 
            "inserted": int(len(saved_ids))
        }
        with open(os.path.join(out_dir, "round_summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        
        print(f"[{round_name}] decisions={len(decisions)} | similar_found={similar_found_count} | "
              f"label_adjusted={label_adjusted_count} | not_found={len(not_found)} | inserted={len(saved_ids)}")
        
        return {
            "decisions": decisions, 
            "inserted": len(saved_ids), 
            "gated": len(packs), 
            "similar_found": similar_found_count,
            "label_adjusted": label_adjusted_count,
            "not_found": len(not_found)
        }

async def main():
    cfg = load_config()
    P  = cfg["paths"]; G = cfg["gating"]; EP = cfg["evidence_pack"]
    R  = cfg["rag"];   D = cfg["decision"];   E = cfg["evidence"]

    predictions_dir = P["predictions_dir"]
    model_path      = P["model_path"]
    evidence_db_dir = P["evidence_db_dir"]
    round_outputs_dir = P.get("round_outputs_dir", "./round_outputs")

    det_tmp = DetectionTool(model_path=model_path, rounds_directory=predictions_dir)
    feature_space = det_tmp.feature_names()
    print(f"[INIT] feature_space = {len(feature_space)} dims")

    rag = FeedbackSearchTool(
        feature_space=feature_space,
        alpha=R.get("alpha", 0.3),           # IP-pair 가중치
        beta=R.get("beta", 0.4),             # Raw feature 가중치
        gamma=R.get("gamma", 0.3),           # SHAP 가중치
        direction_sensitive=R.get("direction_sensitive", True),
        top_k_shap=R.get("top_k_shap", 5),
        rank_diff_weights=R.get("rank_diff_weights", {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}),
        coarse_k=R.get("coarse_k", 200),
        enable_prefilter=R.get("enable_prefilter", True),
    )
    packs_db = load_evidence_packs_from_jsonl(evidence_db_dir, P.get("evidence_packs_file", "packs.jsonl"))
    rag.build(packs_db)
    if len(packs_db) == 0:
        print("[RAG] 첫 실행 감지: Evidence DB가 비었습니다. 이번 라운드 종료 후부터 참조가 활성화됩니다.")

    pri = PrioritizerTool(alpha=G.get("alpha", 0.3), beta=G.get("beta", 0.7),
                          bottom_percent=G.get("bottom_percent", 5.0), top_k=None,
                          out_dir=P.get("feedback_cases_dir", "./feedback_cases"))
    epk = EvidencePackTool(model=det_tmp.model, feature_columns=feature_space,
                           shap_topk=EP.get("shap_topk", 5))
    pol = DecisionPolicy(
        alpha_ip=D.get("alpha_ip", 0.5), 
        beta_cos=D.get("beta_cos", 0.3),
        gamma_overlap=D.get("gamma_overlap", 0.2),
        threshold=D.get("threshold", 0.7), 
        top_refs=D.get("top_refs", 3),
        label_confidence_threshold=D.get("label_confidence_threshold", 0.6)  # ★ 추가
    )
    writer = EvidenceWriter(db_dir=evidence_db_dir,
                            index_file=P.get("evidence_index_file", "evidence_index.jsonl"),
                            packs_file=P.get("evidence_packs_file", "packs.jsonl"),
                            not_found_sim_threshold=E.get("not_found_sim_threshold", 0.15),
                            insertion_rate_percent=E.get("insertion_rate_percent", 3.0),  # ★ 3%
                            dedup_round_digits=E.get("dedup_round_digits", 3))

    orch = FeedNidsOrchestrator(det_tmp, pri, epk, rag, pol, writer)

    _ensure_dir(predictions_dir)
    files = sorted(glob.glob(os.path.join(predictions_dir, "*_with_predictions.csv")))
    if not files:
        print(f"[WARN] 예측 파일 없음: {predictions_dir}")
        print("       *_with_predictions.csv 를 생성한 뒤 실행하세요.")
        return

    _ensure_dir(round_outputs_dir)
    packs_total_after_rebuild = len(packs_db)

    for i, f in enumerate(files, start=1):
        round_name = os.path.basename(f).replace("_with_predictions.csv", "")
        print(f"\n======= [{i}/{len(files)}] Running {round_name} =======")
        this_out = os.path.join(round_outputs_dir, round_name)
        _ensure_dir(this_out)

        result = await orch.handle_predictions_csv(f, round_name, this_out)

        if R.get("reload_after_each_round", True):
            packs_db = load_evidence_packs_from_jsonl(evidence_db_dir, P.get("evidence_packs_file", "packs.jsonl"))
            rag.build(packs_db)
            packs_total_after_rebuild = len(packs_db)
            print(f"[RAG] Rebuilt index with {packs_total_after_rebuild} packs for next rounds.")
            with open(os.path.join(this_out, "RAG_AFTER_REBUILD.txt"), "w", encoding="utf-8") as ftxt:
                ftxt.write(str(packs_total_after_rebuild))

        _append_round_log(os.path.join(round_outputs_dir, "round_log.csv"), {
            "round": round_name,
            "gated": result.get("gated", 0),
            "similar_found": result.get("similar_found", 0),
            "label_adjusted": result.get("label_adjusted", 0),
            "not_found": result.get("not_found", 0),
            "inserted": result.get("inserted", 0),
            "packs_total_after_rebuild": packs_total_after_rebuild
        })

        if cfg.get("pipeline", {}).get("pause_between_rounds", False) and i < len(files):
            try: input("다음 라운드로 진행하려면 Enter...")
            except EOFError: pass

    print("\n[Done] All rounds processed.")

if __name__ == "__main__":
    asyncio.run(main())