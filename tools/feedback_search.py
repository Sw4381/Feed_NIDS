# -*- coding: utf-8 -*-
"""
FeedbackSearchTool - Position-Aware Similarity Search
- 3가지 유사도 결합: IP-pair + Raw Features + Position-Aware SHAP
- S = α·IP_pair + β·cos(raw_features) + γ·Position_Aware_SHAP
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

class FeedbackSearchTool:
    def __init__(self,
                 feature_space: List[str],
                 alpha: float = 0.3,    # IP-pair 가중치
                 beta: float = 0.4,     # Raw feature 가중치
                 gamma: float = 0.3,    # SHAP 가중치
                 direction_sensitive: bool = True,
                 top_k_shap: int = 5,
                 rank_diff_weights: Dict[int, float] = None,
                 coarse_k: int = 200,
                 enable_prefilter: bool = True,
                 # 아래 파라미터들은 호환성 유지용 (사용 안 함)
                 index_type: str = "position_aware",
                 hnsw_M: int = 32,
                 hnsw_efSearch: int = 128,
                 ivf_nlist: int = 8192,
                 ivf_nprobe: int = 32,
                 pq_m: int = 64,
                 pq_bits: int = 8):
        """
        Position-Aware 유사도 검색
        
        Args:
            feature_space: 전체 특징 리스트
            alpha: IP-pair 가중치 (기본 0.3)
            beta: Raw feature cosine 가중치 (기본 0.4)
            gamma: Position-aware SHAP 가중치 (기본 0.3)
            direction_sensitive: 방향 고려 여부 (True: A→B, False: A↔B)
            top_k_shap: SHAP Top-K 개수 (기본 5)
            rank_diff_weights: 순위 차이별 가중치 {0:1.0, 1:0.8, 2:0.6, 3:0.4}
            coarse_k: 반환할 상위 K개
        """
        self.feature_space = feature_space
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.direction_sensitive = direction_sensitive
        self.top_k_shap = top_k_shap
        self.coarse_k = int(coarse_k)
        self.enable_prefilter = bool(enable_prefilter)
        
        # 기본 순위 가중치
        if rank_diff_weights is None:
            self.rank_diff_weights = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        else:
            self.rank_diff_weights = rank_diff_weights
        
        # 내부 상태
        self._packs: List[Dict[str, Any]] = []
        self._preprocessed = False
        
        print(f"[RAG] Position-Aware 유사도 검색 초기화")
        print(f"  가중치: α={alpha} (IP), β={beta} (Raw), γ={gamma} (SHAP)")
        print(f"  순위 가중치: {self.rank_diff_weights}")

    # ------- 유틸리티 -------
    def _extract_feature_vector(self, p: Dict[str, Any]) -> np.ndarray:
        """Evidence Pack에서 전체 특징 벡터 추출"""
        # 1순위: full_features (전체 특징)
        full_feat = p.get("full_features", {})
        if full_feat:
            vals = [float(full_feat.get(feat, 0.0)) for feat in self.feature_space]
            return np.array(vals, dtype=np.float64)
        
        # 2순위: topk_raw (SHAP Top-K만)
        raw_dict = {}
        for item in p.get("topk_raw", []):
            name = item.get("name", "")
            value = item.get("value", 0.0)
            if name:
                raw_dict[name] = float(value)
        
        vals = [raw_dict.get(feat, 0.0) for feat in self.feature_space]
        return np.array(vals, dtype=np.float64)

    def _extract_shap_features(self, p: Dict[str, Any]) -> Dict[str, int]:
        """
        SHAP Top-K 특징과 순위 추출
        
        Returns:
            {feature_name: rank} (rank는 1부터 시작)
        """
        features = {}
        
        # topk_shap에서 추출
        for i, item in enumerate(p.get("topk_shap", [])[:self.top_k_shap], start=1):
            name = item.get("name", "")
            if name and str(name) != 'nan' and str(name) != '':
                features[str(name)] = i
        
        return features

    def _calculate_ip_match(self, query_ip: Dict, evidence_ip: Dict) -> float:
        """IP-pair 일치도 계산 (1.0 or 0.0)"""
        q_src = str(query_ip.get("src", ""))
        q_dst = str(query_ip.get("dst", ""))
        e_src = str(evidence_ip.get("src", ""))
        e_dst = str(evidence_ip.get("dst", ""))
        
        if self.direction_sensitive:
            # 방향 고려: A→B 일치
            return 1.0 if (q_src == e_src and q_dst == e_dst) else 0.0
        else:
            # 방향 무시: {A,B} 일치
            return 1.0 if ({q_src, q_dst} == {e_src, e_dst}) else 0.0

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        코사인 유사도 계산
        -1~1 범위를 0~1로 변환
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        # -1~1 → 0~1 변환
        return max(0.0, min(1.0, (cos_sim + 1) / 2))

    def _calculate_position_aware_overlap(
        self, 
        query_shap: Dict[str, int], 
        evidence_shap: Dict[str, int]
    ) -> Tuple[float, int, List[str]]:
        """
        Position-Aware SHAP 유사도 계산
        
        Returns:
            (overlap_score, common_count, common_features)
        """
        if not query_shap or not evidence_shap:
            return 0.0, 0, []
        
        # 공통 특징 찾기
        common = set(query_shap.keys()) & set(evidence_shap.keys())
        
        if not common:
            return 0.0, 0, []
        
        # 순위 차이 기반 점수 계산
        total_score = 0.0
        for feat in common:
            rank_diff = abs(query_shap[feat] - evidence_shap[feat])
            # 가중치 적용
            total_score += self.rank_diff_weights.get(rank_diff, 0.0)
        
        # 정규화: 최대 점수는 min(len(query), len(evidence))
        max_possible = min(len(query_shap), len(evidence_shap))
        overlap_score = total_score / max_possible if max_possible > 0 else 0.0
        
        return overlap_score, len(common), sorted(list(common))

    # ------- 인덱스 구축 -------
    def build(self, evidence_packs: List[Dict[str, Any]]):
        """Evidence Pack 리스트로 검색 인덱스 구축"""
        self._packs = list(evidence_packs or [])
        
        if len(self._packs) == 0:
            print(f"[RAG] 빈 Evidence DB")
            self._preprocessed = False
            return
        
        print(f"[RAG] Evidence 전처리 중... ({len(self._packs):,}개)")
        
        # 전처리: 특징 벡터 및 SHAP 미리 추출 (캐싱)
        for p in self._packs:
            # 특징 벡터 캐싱
            if '__feature_vector' not in p:
                p['__feature_vector'] = self._extract_feature_vector(p)
            
            # SHAP 특징 캐싱
            if '__shap_features' not in p:
                p['__shap_features'] = self._extract_shap_features(p)
        
        self._preprocessed = True
        print(f"[RAG] 전처리 완료: {len(self._packs):,}개 Evidence")

    # ------- 검색 -------
    def query(self, evidence_pack: Dict[str, Any], k: int = None) -> Dict[str, Any]:
        """
        Position-Aware 유사도 검색
        
        Returns:
            {
                "candidates": [
                    {
                        "ref_id": str,
                        "score": float,  # 총 유사도 점수
                        "label": str,
                        "src": str,
                        "dst": str,
                        "ip_score": float,
                        "cosine_score": float,
                        "shap_score": float,
                        "common_shap_count": int,
                        "common_shap_features": str
                    },
                    ...
                ]
            }
        """
        if k is None:
            k = self.coarse_k
        
        if not self._packs:
            return {"candidates": []}
        
        # Query 특징 추출
        query_vec = self._extract_feature_vector(evidence_pack)
        query_shap = self._extract_shap_features(evidence_pack)
        query_ip = evidence_pack.get("ip_pair", {})
        
        # 모든 Evidence와 유사도 계산
        similarities = []
        
        for p in self._packs:
            # 1. IP-pair 유사도
            evidence_ip = p.get("ip_pair", {})
            ip_score = self._calculate_ip_match(query_ip, evidence_ip)
            
            # 2. Raw Feature 코사인 유사도
            evidence_vec = p.get('__feature_vector')
            if evidence_vec is None:
                evidence_vec = self._extract_feature_vector(p)
            cosine_score = self._calculate_cosine_similarity(query_vec, evidence_vec)
            
            # 3. Position-Aware SHAP 유사도
            evidence_shap = p.get('__shap_features')
            if evidence_shap is None:
                evidence_shap = self._extract_shap_features(p)
            shap_score, common_cnt, common_feats = self._calculate_position_aware_overlap(
                query_shap, evidence_shap
            )
            
            # 4. 총 유사도 = α·IP + β·Cosine + γ·SHAP
            total_score = (
                self.alpha * ip_score +
                self.beta * cosine_score +
                self.gamma * shap_score
            )
            
            similarities.append({
                "pack": p,
                "total_score": total_score,
                "ip_score": ip_score,
                "cosine_score": cosine_score,
                "shap_score": shap_score,
                "common_count": common_cnt,
                "common_features": common_feats
            })
        
        # 유사도 기준 정렬
        similarities.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Top-K 선정
        k = min(k, len(similarities))
        top_k = similarities[:k]
        
        # 결과 포맷팅
        candidates = []
        for sim in top_k:
            p = sim["pack"]
            candidates.append({
                "ref_id": p.get("ref_id", ""),
                "score": float(sim["total_score"]),
                "label": p.get("label", ""),
                "src": (p.get("ip_pair") or {}).get("src", ""),
                "dst": (p.get("ip_pair") or {}).get("dst", ""),
                "ip_score": float(sim["ip_score"]),
                "cosine_score": float(sim["cosine_score"]),
                "shap_score": float(sim["shap_score"]),
                "common_shap_count": int(sim["common_count"]),
                "common_shap_features": ",".join(sim["common_features"])
            })
        
        return {"candidates": candidates}