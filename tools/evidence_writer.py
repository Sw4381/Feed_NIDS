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
    - 라벨은 '모르면 비움("")'을 원칙으로 한다. (Normal 우선 샘플링 없음)
    """
    def __init__(self,
                 db_dir: str = "./evidence_db",
                 index_file: str = "evidence_index.jsonl",
                 packs_file: str = "packs.jsonl",
                 not_found_sim_threshold: float = 0.15,
                 insertion_rate_percent: float = 0.1,
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
    def select_not_found(self, packs: List[Dict[str, Any]], rag_map: Dict[str, Dict[str, Any]]):
        """
        - RAG 결과에서 최고 유사도 < threshold 이거나 후보 없음이면 '검색 안 됨'으로 간주.
        - 반환: DB 기록 후보 목록(오리지널 Pack dict)
        """
        out = []
        for p in packs:
            aid = p.get("alert_id")
            res = rag_map.get(aid, {}) if rag_map else {}
            cands = res.get("candidates", []) or []
            max_sim = max([c.get("score", 0.0) for c in cands], default=0.0)
            if len(cands) == 0 or max_sim < self.not_found_sim_threshold:
                out.append(p)
        return out

    def subsample(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        - 비율 기반 랜덤 샘플링(라벨 우선순위 없음).
        - 최소 1개는 뽑되, 후보가 0이면 빈 리스트.
        """
        if not candidates:
            return []
        take_n = max(1, int(len(candidates) * (self.insertion_rate_percent / 100.0)))
        if take_n >= len(candidates):
            return list(candidates)
        # 안정적 재현성을 위한 고정 시드 기반 표본추출
        idxs = list(range(len(candidates)))
        self.rng.shuffle(idxs)
        pick = [candidates[i] for i in idxs[:take_n]]
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
        - 라벨은 '모르면 비움("")'로 기록(분석자 피드백 있을 때만 들어갈 수 있음).
        - 반환: 저장된 ref_id 목록
        """
        if not packs:
            return []
        saved = []
        idx_rows = []
        pack_rows = []
        for p in packs:
            ref_id = p.get("ref_id") or str(uuid.uuid4())
            # 핵심 필드 구성
            ip = p.get("ip_pair", {})
            topk_shap = p.get("topk_shap", [])
            topk_raw = p.get("topk_raw", [])
            # 라벨: 모르거나 분석 전이면 빈 문자열 유지
            label = p.get("label", "") or p.get("feedback_label", "") or ""

            pack_rows.append({
                "ref_id": ref_id,
                "label": label,
                "ip_pair": {
                    "src": ip.get("src", ""),
                    "dst": ip.get("dst", ""),
                    "proto": ip.get("proto", ""),
                    "dport": ip.get("dport", 0),
                },
                "topk_shap": topk_shap,
                "topk_raw": topk_raw,
                "meta": {
                    "round": p.get("round", ""),
                    "source_alert": p.get("alert_id", ""),
                }
            })
            idx_rows.append({
                "ref_id": ref_id,
                "label": label,
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
        return saved
