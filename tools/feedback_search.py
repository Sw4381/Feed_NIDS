# -*- coding: utf-8 -*-
"""
FeedbackSearchTool
- Evidence Pack들을 벡터화하여 근사 최근접검색(HNSW/IVF-PQ 등) 수행
- 빈 DB에서도 안전 동작, 후보는 ref_id/label/score를 포함
"""
import os
import json
import numpy as np
from typing import List, Dict, Any

class FeedbackSearchTool:
    def __init__(self,
                 feature_space: List[str],
                 index_type: str = "hnsw",
                 hnsw_M: int = 32,
                 hnsw_efSearch: int = 128,
                 ivf_nlist: int = 8192,
                 ivf_nprobe: int = 32,
                 pq_m: int = 64,
                 pq_bits: int = 8,
                 coarse_k: int = 200,
                 enable_prefilter: bool = True):
        self.feature_space = feature_space
        self.index_type = index_type
        self.hnsw_M = hnsw_M
        self.hnsw_efSearch = hnsw_efSearch
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.coarse_k = int(coarse_k)
        self.enable_prefilter = bool(enable_prefilter)
        # 내부 상태
        self._packs: List[Dict[str, Any]] = []
        self._vectors: np.ndarray = None

    # ------- 유틸 -------
    def _pack_to_vector(self, p: Dict[str, Any]) -> np.ndarray:
        """
        Pack의 주요 수치/특징을 고정 길이 벡터로 만든다.
        여기서는 간단히 topk_raw value들만 사용(자유롭게 확장 가능).
        """
        vals = []
        for feat in self.feature_space:
            # feature_space에 맞는 값이 없으면 0.0
            vals.append(0.0)
        # 간단 샘플: topk_raw 값 누적(길이 넘어가면 잘림)
        raw = p.get("topk_raw", [])
        for i, item in enumerate(raw):
            if i >= len(vals): break
            v = item.get("value", 0.0)
            try:
                vals[i] = float(v)
            except Exception:
                vals[i] = 0.0
        return np.asarray(vals, dtype=np.float32)

    # ------- 인덱스 구축 -------
    def build(self, evidence_packs: List[Dict[str, Any]]):
        self._packs = list(evidence_packs or [])
        if len(self._packs) == 0:
            self._vectors = None
            return
        vecs = [self._pack_to_vector(p) for p in self._packs]
        if len(vecs) == 0:
            self._vectors = None
            return
        self._vectors = np.vstack(vecs).astype(np.float32)

    # ------- 질의 -------
    def query(self, evidence_pack: Dict[str, Any], k: int = None) -> Dict[str, Any]:
        if k is None:
            k = self.coarse_k
        if self._vectors is None or len(self._packs) == 0:
            return {"candidates": []}

        q = self._pack_to_vector(evidence_pack)
        # 단순 코사인 유사도
        A = self._vectors
        qn = np.linalg.norm(q) + 1e-9
        An = np.linalg.norm(A, axis=1) + 1e-9
        sims = (A @ q) / (An * qn)
        # Top-k 추출
        k = max(1, min(k, len(self._packs)))
        top_idx = np.argsort(-sims)[:k]

        cands = []
        for idx in top_idx:
            p = self._packs[idx]
            cands.append({
                "ref_id": p.get("ref_id", ""),
                "score": float(sims[idx]),
                "label": p.get("label", ""),  # ★ label 전달(없으면 "")
                "src": (p.get("ip_pair") or {}).get("src", ""),
                "dst": (p.get("ip_pair") or {}).get("dst", ""),
            })
        return {"candidates": cands}
