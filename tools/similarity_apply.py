# -*- coding: utf-8 -*-
"""
Optimized Similarity Apply Tool with Caching
- Annoy 벡터 인덱싱 (극적 성능 개선)
- StandardScaler 정규화 (코사인 유사도 정확도)
- KB 및 Feedback 코퍼스 캐싱 시스템
"""
from __future__ import annotations
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except ImportError:
    HAS_ANNOY = False
    import warnings
    warnings.warn("Annoy 미설치: pip install annoy")

from tools.base import ITool, ToolResult, ensure_dir, get_logger
from cache_manager import CacheManager

log = get_logger("OptimizedSimilarityApplyTool")


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """숫자 피처 컬럼 추출 (라벨/메타데이터 제외)"""
    patterns = [
        r'^label$', r'.*_label$', r'^case_id$', r'^capsule_id$',
        r'.*feedback.*', r'.*applied.*', r'^shap_.*', r'.*reviewed.*',
        r'.*needs_review.*', r'.*review_date.*', r'^A_ip$', r'^B_ip$',
        r'^A_port$', r'^B_port$', r'.*predicted.*', r'.*confidence.*',
        r'.*probability.*', r'.*is_correct.*', r'.*adjusted.*', r'.*reason.*',
        r'.*round.*', r'^capsule_duration$', r'^flow_count$', r'^unique_ports$',
        r'^statistical_score$', r'^__.*', r'.*kb_.*',
    ]
    comp = [re.compile(p, re.IGNORECASE) for p in patterns]
    num = df.select_dtypes(include=['int64', 'float64']).columns
    feats = [c for c in num if not any(p.match(c) for p in comp)]
    return feats


def _shap_rank_map(row: pd.Series, k: int = 5) -> Dict[str, int]:
    """SHAP Top-K 피처 랭킹 추출"""
    d = {}
    for i in range(1, k + 1):
        col = f"shap_top{i}_feature"
        f = row.get(col, "")
        if f and str(f) not in ("", "nan"):
            d[str(f)] = i
    return d


def _rank_overlap(
    t: Dict[str, int],
    f: Dict[str, int],
    weights: Dict[int, float]
) -> Tuple[float, int, List[str]]:
    """SHAP 랭킹 오버랩 계산"""
    if not t or not f:
        return 0.0, 0, []
    com = set(t) & set(f)
    if not com:
        return 0.0, 0, []
    s = 0.0
    for feat in com:
        d = abs(t[feat] - f[feat])
        s += weights.get(d, 0.0)
    mx = min(len(t), len(f))
    return (s / mx if mx > 0 else 0.0), len(com), sorted(list(com))


def _load_feedback_corpus(feedback_dir: str, labels=("Normal", "Attack")) -> pd.DataFrame:
    """Feedback 코퍼스 로드 (reviewed=True만)"""
    paths = sorted(glob.glob(os.path.join(feedback_dir, "*_low_confidence_cases.csv")))
    outs = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        if "reviewed" not in df.columns or "feedback_label" not in df.columns:
            continue
        sub = df[(df["reviewed"] == True) & (df["feedback_label"].isin(labels))].copy()
        if sub.empty:
            continue
        if "feedback_confidence" not in sub.columns:
            sub["feedback_confidence"] = 0
        if "review_date" not in sub.columns:
            sub["review_date"] = ""
        sub["__fb_source"] = os.path.basename(p)
        outs.append(sub)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# ========================================
# Feedback 최적화 (Phase 3)
# ========================================

class OptimizedSimilarityApplyTool(ITool):
    """
    Feedback 기반 최적화된 유사도 적용 (캐싱 포함)
    
    입력: ./feedback_cases/{Round}_low_confidence_cases.csv
    출력: ./round_predictions_applied/{Round}_position_aware_optimal.csv
    """
    
    def __init__(
        self,
        round_name: str,
        feedback_dir: str = "./feedback_cases",
        out_dir: str = "./round_predictions_applied",
        alpha: float = 0.6,
        beta: float = 0.1,
        gamma: float = 0.3,
        threshold: float = 0.6,
        direction_sensitive: bool = True,
        top_k: int = 5,
        rank_weights: Optional[Dict[int, float]] = None,
        n_trees: int = 100,
        annoy_search_k: int = 1000,
        chunk_size: int = 1000,
        cache_dir: str = "./cache",
        use_cache: bool = True,
    ):
        self.round_name = round_name
        self.feedback_dir = feedback_dir
        self.out_dir = out_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.direction_sensitive = direction_sensitive
        self.top_k = top_k
        self.rank_weights = rank_weights or {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        self.n_trees = n_trees
        self.annoy_search_k = annoy_search_k
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        
        # 캐시 매니저
        self.cache_manager = CacheManager(cache_root=cache_dir)
        
        # 런타임 객체
        self.scaler = None
        self.index = None
        self.common_features = []
    
    def _initialize_index_and_scaler(
        self,
        feedback_corpus: pd.DataFrame,
        common_features: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Annoy 인덱스 및 StandardScaler 초기화
        캐시가 있으면 로드, 없으면 빌드 후 저장
        """
        self.common_features = common_features
        n_features = len(common_features)
        
        # 캐시 확인
        if self.use_cache and self.cache_manager.is_fb_cache_valid(
            round_name=self.round_name,
            feedback_corpus=feedback_corpus,
            common_features=common_features,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            n_trees=self.n_trees,
        ):
            # 캐시에서 로드
            (self.scaler, fb_vectors_normalized,
             self.index, metadata) = self.cache_manager.load_fb_cache(
                round_name=self.round_name,
                n_features=n_features,
            )
            log.info(f"✅ [{self.round_name}] Feedback 캐시 사용 (빌드 시간 절약)")
            return fb_vectors_normalized, metadata
        
        # 캐시 없음 → 새로 빌드
        log.info(f"⚙️ [{self.round_name}] Feedback 인덱스 빌드 중...")
        
        # 1. StandardScaler 학습 (Feedback 코퍼스로)
        log.info(f"  [1/3] StandardScaler 초기화 (Feedback 코퍼스로 학습)...")
        self.scaler = StandardScaler()
        fb_features = feedback_corpus[common_features].values.astype(np.float64)
        fb_features = np.nan_to_num(fb_features, nan=0.0)
        self.scaler.fit(fb_features)
        log.info(f"  StandardScaler 학습 완료: {len(feedback_corpus)} 케이스, {n_features} features")
        
        # 2. Feedback 벡터 정규화
        log.info(f"  [2/3] Feedback 벡터 정규화 중...")
        fb_vectors_normalized = self.scaler.transform(fb_features)
        log.info(f"  정규화 완료: {fb_vectors_normalized.shape}")
        
        # 3. Annoy 인덱스 빌드
        log.info(f"  [3/3] Annoy 인덱스 빌드 중 (n_trees={self.n_trees})...")
        self.index = AnnoyIndex(n_features, 'angular')
        for i, vec in enumerate(fb_vectors_normalized):
            self.index.add_item(i, vec)
            if (i + 1) % 10000 == 0:
                log.info(f"    {i+1}/{len(fb_vectors_normalized)} 추가됨...")
        
        self.index.build(self.n_trees)
        log.info(f"  Annoy 인덱스 빌드 완료: {self.index.get_n_items()} items")
        
        # 4. 메타데이터 준비
        metadata = {
            "case_ids": feedback_corpus["case_id"].tolist() if "case_id" in feedback_corpus.columns else [],
            "feedback_labels": feedback_corpus["feedback_label"].tolist() if "feedback_label" in feedback_corpus.columns else [],
            "feedback_confidence": feedback_corpus["feedback_confidence"].tolist() if "feedback_confidence" in feedback_corpus.columns else [],
            "feedback_reason": feedback_corpus["feedback_reason"].tolist() if "feedback_reason" in feedback_corpus.columns else [],
            "fb_source": feedback_corpus["__fb_source"].tolist() if "__fb_source" in feedback_corpus.columns else [],
            "A_ips": feedback_corpus["A_ip"].tolist() if "A_ip" in feedback_corpus.columns else [],
            "B_ips": feedback_corpus["B_ip"].tolist() if "B_ip" in feedback_corpus.columns else [],
            "shap_features": [_shap_rank_map(row, self.top_k) for _, row in feedback_corpus.iterrows()] if self.gamma > 0 else [],
        }
        
        # 5. 캐시 저장
        if self.use_cache:
            self.cache_manager.save_fb_cache(
                round_name=self.round_name,
                scaler=self.scaler,
                fb_vectors_normalized=fb_vectors_normalized,
                annoy_index=self.index,
                metadata=metadata,
                feedback_corpus=feedback_corpus,
                common_features=common_features,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                n_trees=self.n_trees,
            )
        
        return fb_vectors_normalized, metadata
    
    def run(self) -> ToolResult:
        """메인 실행 로직"""
        if not HAS_ANNOY:
            log.warning("Annoy 미설치 → 기본 SimilarityApplyTool 사용 권장")
            return ToolResult(False, "Annoy 미설치: pip install annoy")
        
        ensure_dir(self.out_dir)
        
        # 타겟 파일 로드
        pred_file = os.path.join(self.feedback_dir, f"{self.round_name}_low_confidence_cases.csv")
        if not os.path.exists(pred_file):
            return ToolResult(False, f"타겟 파일 없음: {pred_file}")
        
        df = pd.read_csv(pred_file, low_memory=False)
        feat_cols = _get_feature_columns(df)
        
        # SHAP 컬럼 검증
        gamma = self.gamma
        if gamma > 0:
            miss = [
                f"shap_top{i}_feature" for i in range(1, self.top_k + 1)
                if f"shap_top{i}_feature" not in df.columns
            ]
            if miss:
                log.warning(f"SHAP 컬럼 누락 → gamma=0으로 계속: {miss}")
                gamma = 0.0
        
        # Feedback 코퍼스 로드
        fb = _load_feedback_corpus(self.feedback_dir, ("Normal", "Attack"))
        if fb.empty:
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(True, "코퍼스 없음 → 원본 저장", output_path=out_path)
        
        # 공통 피처
        fb_cols = _get_feature_columns(fb)
        common = sorted(list(set(feat_cols) & set(fb_cols)))
        if not common:
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(True, "공통 피처 없음 → 원본 저장", output_path=out_path)
        
        log.info(f"공통 피처: {len(common)}개")
        
        # 인덱스 초기화 (캐싱 활용)
        fb_vectors_normalized, metadata = self._initialize_index_and_scaler(fb, common)
        
        # 출력 필드 준비
        if "adjusted_label" not in df.columns:
            df["adjusted_label"] = df.get("predicted_label", "")
        for c in ["feedback_applied", "applied_from_case", "applied_from_file",
                  "applied_reason", "applied_confidence", "applied_common_features"]:
            if c not in df.columns:
                df[c] = False if c == "feedback_applied" else ""
        for c in ["applied_similarity_score", "applied_similarity_ip",
                  "applied_similarity_cosine", "applied_similarity_overlap"]:
            if c not in df.columns:
                df[c] = 0.0
        
        # 본 처리 (2단계 필터링)
        applied = 0
        total = len(df)
        
        for s in range(0, total, self.chunk_size):
            e = min(s + self.chunk_size, total)
            log.info(f"{self.round_name} 진행 {e}/{total} ({e/total*100:.1f}%)")
            
            for idx, row in df.iloc[s:e].iterrows():
                # 타겟 벡터 정규화
                t_vec_raw = np.nan_to_num(row[common].values.astype(np.float64), nan=0.0)
                t_vec = self.scaler.transform([t_vec_raw])[0]
                
                t_shap = _shap_rank_map(row, self.top_k) if gamma > 0 else {}
                ta, tb = str(row.get("A_ip", "")), str(row.get("B_ip", ""))
                
                # Stage 1: Annoy로 Top-K 추출 (코사인 기반)
                neighbors, cosines = self.index.get_nns_by_vector(
                    t_vec, self.annoy_search_k, include_distances=True
                )
                
                # Stage 2: Top-K 내에서 정확한 S 스코어 계산
                best_score, best_idx, comp = 0.0, None, None
                
                for fb_idx, cos_dist in zip(neighbors, cosines):
                    # IP 유사도
                    fb_a = metadata["A_ips"][fb_idx]
                    fb_b = metadata["B_ips"][fb_idx]
                    
                    if self.direction_sensitive:
                        ip = 1.0 if (ta == str(fb_a) and tb == str(fb_b)) else 0.0
                    else:
                        ip = 1.0 if {ta, tb} == {str(fb_a), str(fb_b)} else 0.0
                    
                    # 코사인 유사도 (Annoy는 angular distance 반환)
                    cs = max(0.0, 1.0 - cos_dist / 2.0)
                    
                    # SHAP 오버랩
                    if gamma > 0 and metadata["shap_features"]:
                        fb_shap = metadata["shap_features"][fb_idx]
                        ov, cc, com_feats = _rank_overlap(t_shap, fb_shap, self.rank_weights)
                    else:
                        ov, cc, com_feats = 0.0, 0, []
                    
                    # 종합 스코어
                    score = self.alpha * ip + self.beta * cs + gamma * ov
                    
                    if score > best_score:
                        best_score = score
                        best_idx = fb_idx
                        comp = {
                            "ip": ip,
                            "cos": cs,
                            "ov": ov,
                            "cc": cc,
                            "com": ",".join(com_feats) if com_feats else ""
                        }
                
                # 임계값 이상이면 적용
                if best_idx is not None and best_score >= self.threshold:
                    df.at[idx, "adjusted_label"] = metadata["feedback_labels"][best_idx]
                    df.at[idx, "feedback_applied"] = True
                    df.at[idx, "applied_from_case"] = str(metadata["case_ids"][best_idx])
                    df.at[idx, "applied_from_file"] = str(metadata["fb_source"][best_idx])
                    df.at[idx, "applied_reason"] = str(metadata["feedback_reason"][best_idx])
                    df.at[idx, "applied_confidence"] = str(metadata["feedback_confidence"][best_idx])
                    df.at[idx, "applied_similarity_score"] = best_score
                    df.at[idx, "applied_similarity_ip"] = comp["ip"]
                    df.at[idx, "applied_similarity_cosine"] = comp["cos"]
                    df.at[idx, "applied_similarity_overlap"] = comp["ov"]
                    df.at[idx, "applied_common_features"] = comp["com"]
                    applied += 1
        
        # 결과 저장
        out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"유사도 적용 {applied}/{len(df)} → {out_path}")
        
        return ToolResult(
            True, "ok",
            output_path=out_path,
            data={"applied": applied, "total": len(df)}
        )


# ========================================
# KB 최적화 (Phase 2)
# ========================================

class OptimizedKBSimilarityApplyTool(ITool):
    """
    Knowledge Base 기반 최적화된 유사도 적용 (캐싱 포함)
    
    입력: 
    - ./round_predictions/{Round}_with_predictions.csv
    - kb_corpus: Knowledge Base DataFrame
    
    출력:
    - ./round_predictions/{Round}_kb_applied.csv
    """
    
    def __init__(
        self,
        round_name: str,
        pred_dir: str = "./round_predictions",
        kb_corpus: pd.DataFrame = None,
        out_dir: str = "./round_predictions",
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.3,
        threshold: float = 0.6,
        direction_sensitive: bool = True,
        top_k: int = 5,
        rank_weights: Optional[Dict[int, float]] = None,
        n_trees: int = 100,
        annoy_search_k: int = 1000,
        chunk_size: int = 10000,
        cache_dir: str = "./cache",
        use_cache: bool = True,
    ):
        self.round_name = round_name
        self.pred_dir = pred_dir
        self.kb_corpus = kb_corpus
        self.out_dir = out_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.direction_sensitive = direction_sensitive
        self.top_k = top_k
        self.rank_weights = rank_weights or {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        self.n_trees = n_trees
        self.annoy_search_k = annoy_search_k
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        
        # 캐시 매니저
        self.cache_manager = CacheManager(cache_root=cache_dir)
        
        # 런타임 객체
        self.scaler = None
        self.index = None
        self.common_features = []
    
    def _initialize_kb_index_and_scaler(
        self,
        kb_corpus: pd.DataFrame,
        common_features: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        KB Annoy 인덱스 및 StandardScaler 초기화
        캐시가 있으면 로드, 없으면 빌드 후 저장
        """
        self.common_features = common_features
        n_features = len(common_features)
        
        # 캐시 확인
        if self.use_cache and self.cache_manager.is_kb_cache_valid(
            kb_corpus=kb_corpus,
            common_features=common_features,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            n_trees=self.n_trees,
        ):
            # 캐시에서 로드
            (self.scaler, kb_vectors_normalized,
             self.index, metadata) = self.cache_manager.load_kb_cache(
                n_features=n_features,
            )
            log.info(f"✅ KB 캐시 사용 (빌드 시간 절약: ~40분)")
            return kb_vectors_normalized, metadata
        
        # 캐시 없음 → 새로 빌드
        log.info("⚙️ KB 기반 인덱스 및 스케일러 초기화...")
        
        # 1. StandardScaler 학습 (KB로)
        log.info(f"  [1/3] StandardScaler 초기화 (KB로 학습)...")
        self.scaler = StandardScaler()
        kb_features = kb_corpus[common_features].values.astype(np.float64)
        kb_features = np.nan_to_num(kb_features, nan=0.0)
        
        log.info(f"  StandardScaler 학습: {len(kb_corpus)} 케이스, {n_features} features")
        self.scaler.fit(kb_features)
        log.info("  StandardScaler 학습 완료")
        log.info(f"    Mean: {self.scaler.mean_[:3]}... (처음 3개)")
        log.info(f"    Scale: {self.scaler.scale_[:3]}... (처음 3개)")
        
        # 2. KB 벡터 정규화
        log.info(f"  [2/3] KB 벡터 생성 중...")
        kb_vectors_normalized = self.scaler.transform(kb_features)
        log.info(f"  정규화 완료: {kb_vectors_normalized.shape}")
        
        # 3. Annoy 인덱스 빌드
        log.info(f"  [3/3] Annoy 인덱스 빌드 중 (n_trees={self.n_trees})...")
        self.index = AnnoyIndex(n_features, 'angular')
        
        for i, vec in enumerate(kb_vectors_normalized):
            self.index.add_item(i, vec)
            if (i + 1) % 100000 == 0:
                log.info(f"    {i+1}/{len(kb_vectors_normalized)} 벡터 추가됨...")
        
        log.info(f"  Annoy 트리 빌드 시작 (n_trees={self.n_trees})...")
        self.index.build(self.n_trees)
        log.info(f"  Annoy 인덱스 빌드 완료: {self.index.get_n_items()} items")
        
        # 4. 메타데이터 준비
        metadata = {
            "case_ids": kb_corpus["case_id"].tolist() if "case_id" in kb_corpus.columns else list(range(len(kb_corpus))),
            "labels": kb_corpus["label"].tolist() if "label" in kb_corpus.columns else [],
            "A_ips": kb_corpus["A_ip"].tolist() if "A_ip" in kb_corpus.columns else [],
            "B_ips": kb_corpus["B_ip"].tolist() if "B_ip" in kb_corpus.columns else [],
            "shap_features": [_shap_rank_map(row, self.top_k) for _, row in kb_corpus.iterrows()] if self.gamma > 0 else [],
        }
        
        # 5. 캐시 저장
        if self.use_cache:
            self.cache_manager.save_kb_cache(
                scaler=self.scaler,
                kb_vectors_normalized=kb_vectors_normalized,
                annoy_index=self.index,
                metadata=metadata,
                kb_corpus=kb_corpus,
                common_features=common_features,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                n_trees=self.n_trees,
            )
        
        return kb_vectors_normalized, metadata
    
    def run(self) -> ToolResult:
        """메인 실행 로직"""
        if not HAS_ANNOY:
            log.warning("Annoy 미설치 → 기본 KBSimilarityApplyTool 사용 권장")
            return ToolResult(False, "Annoy 미설치: pip install annoy")
        
        ensure_dir(self.out_dir)
        
        # KB 검증
        if self.kb_corpus is None or self.kb_corpus.empty:
            log.warning(f"[{self.round_name}] Knowledge Base 없음 → 스킵")
            return ToolResult(True, "kb_empty", output_path=None)
        
        # 예측 파일 로드
        pred_file = os.path.join(self.pred_dir, f"{self.round_name}_with_predictions.csv")
        if not os.path.exists(pred_file):
            return ToolResult(False, f"예측 파일 없음: {pred_file}")
        
        log.info(f"예측 데이터 로드: {pred_file}")
        df = pd.read_csv(pred_file, low_memory=False)
        log.info(f"  {len(df)} 케이스 로드됨")
        
        feat_cols = _get_feature_columns(df)
        
        # SHAP 컬럼 검증
        gamma = self.gamma
        if gamma > 0:
            miss = [
                f"shap_top{i}_feature" for i in range(1, self.top_k + 1)
                if f"shap_top{i}_feature" not in df.columns
            ]
            if miss:
                log.warning(f"SHAP 컬럼 누락 → gamma=0으로 계속: {miss}")
                gamma = 0.0
        
        # KB 피처 추출
        kb_cols = _get_feature_columns(self.kb_corpus)
        common = sorted(list(set(feat_cols) & set(kb_cols)))
        if not common:
            log.warning(f"[{self.round_name}] 공통 피처 없음")
            out_path = os.path.join(self.out_dir, f"{self.round_name}_kb_applied.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(True, "no_common_features", output_path=out_path)
        
        log.info(f"공통 피처: {len(common)}개")
        
        # 인덱스 초기화 (캐싱 활용)
        log.info("인덱스 및 스케일러 초기화...")
        kb_vectors_normalized, metadata = self._initialize_kb_index_and_scaler(
            self.kb_corpus, common
        )
        
        # 출력 필드 준비
        if "kb_applied_label" not in df.columns:
            df["kb_applied_label"] = ""
        if "kb_applied" not in df.columns:
            df["kb_applied"] = False
        if "kb_from_case" not in df.columns:
            df["kb_from_case"] = ""
        if "kb_similarity_score" not in df.columns:
            df["kb_similarity_score"] = 0.0
        if "kb_similarity_ip" not in df.columns:
            df["kb_similarity_ip"] = 0.0
        if "kb_similarity_cosine" not in df.columns:
            df["kb_similarity_cosine"] = 0.0
        if "kb_similarity_overlap" not in df.columns:
            df["kb_similarity_overlap"] = 0.0
        if "kb_common_features" not in df.columns:
            df["kb_common_features"] = ""
        
        # 본 처리 (2단계 필터링)
        applied = 0
        total = len(df)
        
        log.info(f"KB 기반 유사도 검색 시작 ({total} 케이스)")
        
        for s in range(0, total, self.chunk_size):
            e = min(s + self.chunk_size, total)
            log.info(f"[{self.round_name}] KB 적용 진행 {e}/{total} ({e/total*100:.1f}%)")
            
            for idx, row in df.iloc[s:e].iterrows():
                # 타겟 벡터 정규화
                t_vec_raw = np.nan_to_num(row[common].values.astype(np.float64), nan=0.0)
                t_vec = self.scaler.transform([t_vec_raw])[0]
                
                t_shap = _shap_rank_map(row, self.top_k) if gamma > 0 else {}
                ta = str(row.get("A_ip", ""))
                tb = str(row.get("B_ip", ""))
                
                # Stage 1: Annoy로 Top-K 추출
                neighbors, cosines = self.index.get_nns_by_vector(
                    t_vec, self.annoy_search_k, include_distances=True
                )
                
                # Stage 2: 정확한 S 스코어 계산
                best_score = 0.0
                best_idx = None
                comp = None
                
                for kb_idx, cos_dist in zip(neighbors, cosines):
                    # IP 유사도
                    kb_a = metadata["A_ips"][kb_idx]
                    kb_b = metadata["B_ips"][kb_idx]
                    
                    if self.direction_sensitive:
                        ip = 1.0 if (ta == str(kb_a) and tb == str(kb_b)) else 0.0
                    else:
                        ip = 1.0 if {ta, tb} == {str(kb_a), str(kb_b)} else 0.0
                    
                    # 코사인 유사도
                    cs = max(0.0, 1.0 - cos_dist / 2.0)
                    
                    # SHAP 오버랩
                    if gamma > 0 and metadata["shap_features"]:
                        kb_shap = metadata["shap_features"][kb_idx]
                        ov, cc, com_feats = _rank_overlap(t_shap, kb_shap, self.rank_weights)
                    else:
                        ov, cc, com_feats = 0.0, 0, []
                    
                    # 종합 스코어
                    score = self.alpha * ip + self.beta * cs + gamma * ov
                    
                    if score > best_score:
                        best_score = score
                        best_idx = kb_idx
                        comp = {
                            "ip": ip,
                            "cos": cs,
                            "ov": ov,
                            "cc": cc,
                            "com": ",".join(com_feats) if com_feats else ""
                        }
                
                # 임계값 이상이면 적용
                if best_idx is not None and best_score >= self.threshold:
                    df.at[idx, "kb_applied_label"] = metadata["labels"][best_idx]
                    df.at[idx, "kb_applied"] = True
                    df.at[idx, "kb_from_case"] = str(metadata["case_ids"][best_idx])
                    df.at[idx, "kb_similarity_score"] = best_score
                    df.at[idx, "kb_similarity_ip"] = comp["ip"]
                    df.at[idx, "kb_similarity_cosine"] = comp["cos"]
                    df.at[idx, "kb_similarity_overlap"] = comp["ov"]
                    df.at[idx, "kb_common_features"] = comp["com"]
                    applied += 1
        
        # 결과 저장
        out_path = os.path.join(self.out_dir, f"{self.round_name}_kb_applied.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"[{self.round_name}] KB 기반 적용 완료: {applied}/{total} 케이스 조정")
        
        return ToolResult(
            True, "ok",
            output_path=out_path,
            data={"applied": applied, "total": total, "threshold": self.threshold}
        )