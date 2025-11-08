# -*- coding: utf-8 -*-
import os
import json
import uuid
import random
from pathlib import Path
from typing import List, Dict, Any

class EvidenceWriter:
    """
    - Evidence DB에 신규 Evidence Pack을 기록하고, 인덱스 파일에 요약을 남긴다.
    - Label을 포함하여 저장 (adjusted_label 사용)
    """
    def __init__(self,
                 db_dir: str = "./evidence_db",
                 index_file: str = "evidence_index.jsonl",
                 packs_file: str = "packs.jsonl",
                 not_found_sim_threshold: float = 0.15,
                 insertion_rate_percent: float = 3.0,  # ★ 3%로 변경
                 dedup_round_digits: int = 3,
                 random_seed: int = 42):
        self.db_dir = db_dir
        self.index_path = os.path.join(db_dir, index_file)
        self.packs_path = os.path.join(db_dir, packs_file)
        self.not_found_sim_threshold = float(not_found_sim_threshold)
        self.insertion_rate_percent = float(insertion_rate_percent)
        self.dedup_round_digits = int(dedup_round_digits)
        self.rng = random.Random(random_seed)
        Path(self.db_dir).mkdir(parents=True, exist_ok=True)

    # ========= 셀렉션 =========
    def select_not_found(self, packs: List[Dict[str, Any]], rag_map: Dict[str, Dict[str, Any]], 
                        decisions: List[Dict[str, Any]] = None):
        """
        Evidence 저장 대상 선정:
        1. 유사사례 없음 (RAG 검색 실패)
        2. 유사사례는 있지만 Label 재조정 안 됨
           - NO_LABELED_CASE: 유사사례에 Label 없음
           - LOW_SIMILARITY: 유사도 부족
           - REFER_TO_SIMILAR: 참고만, 확정 안 됨
        
        반환: DB 기록 후보 목록(Pack dict + decision info)
        """
        # Decision Map 생성 (alert_id -> decision info)
        decision_map = {}
        if decisions:
            for d in decisions:
                aid = d.get("alert_id")
                # ★ Ground Truth를 저장용 Label로 사용
                ground_truth = d.get("ground_truth_label", "")
                adjusted = d.get("adjusted_label", "")
                
                decision_map[aid] = {
                    "ground_truth_label": ground_truth,  # ★ 실제 정답
                    "adjusted_label": adjusted,           # 재조정된 예측
                    "confidence": d.get("confidence", 0.0),
                    "decision": d.get("decision", ""),
                    "reasoning": d.get("reasoning", "")
                }
        
        # Evidence 저장 대상이 되는 Decision 타입
        UNCERTAIN_DECISIONS = {
            "NO_SIMILAR_CASE",      # 유사사례 없음
            "NO_LABELED_CASE",      # 유사사례에 Label 없음
            "LOW_SIMILARITY",       # 유사도 부족
            "REFER_TO_SIMILAR"      # 참고만, 확정 안 됨
        }
        
        out = []
        for p in packs:
            aid = p.get("alert_id")
            res = rag_map.get(aid, {}) if rag_map else {}
            cands = res.get("candidates", []) or []
            max_sim = max([c.get("score", 0.0) for c in cands], default=0.0)
            
            # Decision 정보 가져오기
            dec_info = decision_map.get(aid, {})
            decision_type = dec_info.get("decision", "")
            
            # 선정 조건:
            # 1) 유사사례 없음 (기존 로직)
            # 2) Decision이 불확실한 경우 (신규)
            should_include = False
            
            if len(cands) == 0 or max_sim < self.not_found_sim_threshold:
                should_include = True
                reason = "no_similar_case"
            elif decision_type in UNCERTAIN_DECISIONS:
                should_include = True
                reason = "uncertain_decision"
            
            if should_include:
                # Pack에 decision 정보 추가
                p_copy = p.copy()
                if aid in decision_map:
                    # ★ Ground Truth를 주 Label로 사용
                    ground_truth = dec_info.get("ground_truth_label", "")
                    if not ground_truth:
                        # Ground Truth 없으면 adjusted 사용 (fallback)
                        ground_truth = dec_info.get("adjusted_label", "") or "Unknown"
                    
                    p_copy.update(dec_info)
                    p_copy["label"] = ground_truth  # ★ Evidence 저장용 Label = Ground Truth
                else:
                    # Decision이 없는 경우
                    p_copy["ground_truth_label"] = "Unknown"
                    p_copy["adjusted_label"] = "Unknown"
                    p_copy["label"] = "Unknown"
                    p_copy["confidence"] = 0.0
                    p_copy["decision"] = "UNKNOWN"
                
                p_copy["evidence_reason"] = reason  # 디버깅용
                out.append(p_copy)
        
        return out

    def subsample(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        - 비율 기반 랜덤 샘플링 (하위 3%)
        - 최소 1개는 뽑되, 후보가 0이면 빈 리스트.
        - confidence 기준으로 정렬하여 하위부터 샘플링
        """
        if not candidates:
            return []
        
        # confidence 기준 오름차순 정렬 (낮은 confidence = 불확실한 케이스 우선)
        sorted_candidates = sorted(candidates, key=lambda x: x.get("confidence", 0.0))
        
        take_n = max(1, int(len(sorted_candidates) * (self.insertion_rate_percent / 100.0)))
        if take_n >= len(sorted_candidates):
            return list(sorted_candidates)
        
        # 하위 3%를 선택
        pick = sorted_candidates[:take_n]
        return pick

    # ========= I/O =========
    def _append_jsonl(self, path: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def write(self, packs: List[Dict[str, Any]]) -> List[str]:
        """
        - packs를 Evidence DB에 기록한다.
        - ref_id를 새로 부여하고 packs.jsonl과 index.jsonl에 함께 남긴다.
        - ★ Ground Truth Label을 저장 (유사사례 검색 시 활용)
        - 반환: 저장된 ref_id 목록
        """
        if not packs:
            return []
        
        # ★ Label 통계 계산
        label_stats = {}
        for p in packs:
            # label이 이미 select_not_found에서 설정됨 (Ground Truth)
            lbl = p.get("label", "") or ""
            label_stats[lbl if lbl else "(empty)"] = label_stats.get(lbl if lbl else "(empty)", 0) + 1
        
        saved = []
        idx_rows = []
        pack_rows = []
        for p in packs:
            ref_id = p.get("ref_id") or str(uuid.uuid4())
            # 핵심 필드 구성
            ip = p.get("ip_pair", {})
            topk_shap = p.get("topk_shap", [])
            topk_raw = p.get("topk_raw", [])
            
            # ★ label 필드 사용 (Ground Truth가 이미 설정됨)
            label = p.get("label", "") or ""

            pack_rows.append({
                "ref_id": ref_id,
                "label": label,  # ★ Ground Truth Label 저장
                "ip_pair": {
                    "src": ip.get("src", ""),
                    "dst": ip.get("dst", ""),
                    "proto": ip.get("proto", ""),
                    "dport": ip.get("dport", 0),
                },
                "topk_shap": topk_shap,
                "topk_raw": topk_raw,
                "meta": {
                    "round": p.get("meta", {}).get("round", "") or p.get("round", ""),
                    "source_alert": p.get("alert_id", ""),
                    "confidence": p.get("confidence", 0.0),
                    "decision": p.get("decision", ""),
                    "reasoning": p.get("reasoning", ""),
                    "ground_truth_label": p.get("ground_truth_label", ""),
                    "adjusted_label": p.get("adjusted_label", "")
                }
            })
            idx_rows.append({
                "ref_id": ref_id,
                "label": label,  # ★ Ground Truth Label 저장
                "src": ip.get("src", ""),
                "dst": ip.get("dst", ""),
                "proto": ip.get("proto", ""),
                "dport": ip.get("dport", 0),
            })
            saved.append(ref_id)

        Path(self.db_dir).mkdir(parents=True, exist_ok=True)
        # 파일에 추가
        self._append_jsonl(self.packs_path, pack_rows)
        self._append_jsonl(self.index_path, idx_rows)
        
        # ★ 상세 통계 출력
        print(f"[EvidenceWriter] {len(saved)}개 Evidence 저장 완료")
        print(f"  Label 분포 (Ground Truth): {label_stats}")
        return saved