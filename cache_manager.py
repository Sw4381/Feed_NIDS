# -*- coding: utf-8 -*-
"""
Cache Manager for FEED-NIDS Pipeline
KB 인덱스 및 Feedback 코퍼스 캐싱 시스템
"""
from __future__ import annotations
import os
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except ImportError:
    HAS_ANNOY = False

from tools.base import get_logger, ensure_dir

log = get_logger("CacheManager")


class CacheManager:
    """
    캐싱 시스템 관리자
    - KB 인덱스 캐싱
    - Feedback 코퍼스 캐싱
    - 캐시 유효성 검증
    """
    
    def __init__(self, cache_root: str = "./cache"):
        self.cache_root = cache_root
        self.kb_cache_dir = os.path.join(cache_root, "kb_index")
        self.fb_cache_dir = os.path.join(cache_root, "feedback_corpus")
        
    # ========================================
    # KB 인덱스 캐싱
    # ========================================
    
    def get_kb_cache_paths(self) -> Dict[str, str]:
        """KB 캐시 파일 경로 반환"""
        return {
            "scaler": os.path.join(self.kb_cache_dir, "kb_scaler.pkl"),
            "vectors": os.path.join(self.kb_cache_dir, "kb_vectors.npy"),
            "annoy": os.path.join(self.kb_cache_dir, "kb_annoy_index.ann"),
            "metadata": os.path.join(self.kb_cache_dir, "kb_metadata.pkl"),
            "info": os.path.join(self.kb_cache_dir, "kb_cache_info.json"),
        }
    
    def compute_kb_hash(self, kb_corpus: pd.DataFrame) -> str:
        """KB 데이터 해시 계산 (빠른 검증용)"""
        # KB 크기 + 첫/마지막 10개 case_id + label 분포로 해시
        hash_input = f"{len(kb_corpus)}"
        
        if "case_id" in kb_corpus.columns:
            first_ids = kb_corpus["case_id"].head(10).astype(str).tolist()
            last_ids = kb_corpus["case_id"].tail(10).astype(str).tolist()
            hash_input += "_" + "_".join(first_ids + last_ids)
        
        if "label" in kb_corpus.columns:
            label_dist = kb_corpus["label"].value_counts().to_dict()
            hash_input += "_" + str(sorted(label_dist.items()))
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def is_kb_cache_valid(
        self,
        kb_corpus: pd.DataFrame,
        common_features: list,
        alpha: float,
        beta: float,
        gamma: float,
        n_trees: int,
    ) -> bool:
        """KB 캐시 유효성 검증"""
        paths = self.get_kb_cache_paths()
        
        # 1. 파일 존재 여부
        if not all(os.path.exists(p) for p in paths.values()):
            log.info("❌ 캐시 파일 없음")
            return False
        
        # 2. 캐시 정보 로드
        try:
            with open(paths["info"], "r", encoding="utf-8") as f:
                cache_info = json.load(f)
        except Exception as e:
            log.warning(f"❌ 캐시 정보 로드 실패: {e}")
            return False
        
        # 3. KB 크기 일치
        if cache_info.get("kb_size") != len(kb_corpus):
            log.info(f"❌ KB 크기 불일치: 캐시={cache_info.get('kb_size')}, 현재={len(kb_corpus)}")
            return False
        
        # 4. 피처 수 일치
        if cache_info.get("n_features") != len(common_features):
            log.info(f"❌ 피처 수 불일치: 캐시={cache_info.get('n_features')}, 현재={len(common_features)}")
            return False
        
        # 5. 파라미터 일치
        if (cache_info.get("alpha") != alpha or
            cache_info.get("beta") != beta or
            cache_info.get("gamma") != gamma or
            cache_info.get("n_trees") != n_trees):
            log.info("❌ 파라미터 불일치")
            return False
        
        # 6. 데이터 해시 일치 (선택적)
        current_hash = self.compute_kb_hash(kb_corpus)
        if cache_info.get("kb_hash") != current_hash:
            log.info("❌ KB 데이터 해시 불일치 (내용 변경됨)")
            return False
        
        log.info("✅ KB 캐시 유효성 검증 통과")
        return True
    
    def load_kb_cache(
        self,
        n_features: int,
    ) -> Tuple[object, np.ndarray, AnnoyIndex, Dict[str, Any]]:
        """
        KB 캐시 로드
        Returns: (scaler, kb_vectors_normalized, annoy_index, metadata)
        """
        log.info("📂 KB 캐시 로드 중...")
        paths = self.get_kb_cache_paths()
        
        # 1. Scaler 로드
        with open(paths["scaler"], "rb") as f:
            scaler = pickle.load(f)
        log.info("  ✅ Scaler 로드 완료")
        
        # 2. 벡터 로드
        kb_vectors_normalized = np.load(paths["vectors"])
        log.info(f"  ✅ KB 벡터 로드 완료: {kb_vectors_normalized.shape}")
        
        # 3. Annoy 인덱스 로드
        if not HAS_ANNOY:
            raise ImportError("Annoy가 설치되지 않았습니다: pip install annoy")
        
        annoy_index = AnnoyIndex(n_features, 'angular')
        annoy_index.load(paths["annoy"])
        log.info(f"  ✅ Annoy 인덱스 로드 완료: {annoy_index.get_n_items()} items")
        
        # 4. 메타데이터 로드
        with open(paths["metadata"], "rb") as f:
            metadata = pickle.load(f)
        log.info("  ✅ 메타데이터 로드 완료")
        
        log.info("✅ KB 캐시 로드 완료 (약 5-10초)")
        return scaler, kb_vectors_normalized, annoy_index, metadata
    
    def save_kb_cache(
        self,
        scaler: object,
        kb_vectors_normalized: np.ndarray,
        annoy_index: AnnoyIndex,
        metadata: Dict[str, Any],
        kb_corpus: pd.DataFrame,
        common_features: list,
        alpha: float,
        beta: float,
        gamma: float,
        n_trees: int,
    ) -> None:
        """KB 캐시 저장"""
        log.info("💾 KB 캐시 저장 중...")
        ensure_dir(self.kb_cache_dir)
        paths = self.get_kb_cache_paths()
        
        # 1. Scaler 저장
        with open(paths["scaler"], "wb") as f:
            pickle.dump(scaler, f)
        log.info("  ✅ Scaler 저장 완료")
        
        # 2. 벡터 저장
        np.save(paths["vectors"], kb_vectors_normalized)
        log.info(f"  ✅ KB 벡터 저장 완료: {kb_vectors_normalized.shape}")
        
        # 3. Annoy 인덱스 저장
        annoy_index.save(paths["annoy"])
        log.info(f"  ✅ Annoy 인덱스 저장 완료")
        
        # 4. 메타데이터 저장
        with open(paths["metadata"], "wb") as f:
            pickle.dump(metadata, f)
        log.info("  ✅ 메타데이터 저장 완료")
        
        # 5. 캐시 정보 저장
        cache_info = {
            "kb_size": len(kb_corpus),
            "n_features": len(common_features),
            "kb_hash": self.compute_kb_hash(kb_corpus),
            "created_at": datetime.now().isoformat(),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n_trees": n_trees,
            "cache_version": "1.0",
        }
        with open(paths["info"], "w", encoding="utf-8") as f:
            json.dump(cache_info, f, indent=2, ensure_ascii=False)
        log.info("  ✅ 캐시 정보 저장 완료")
        
        log.info("✅ KB 캐시 저장 완료")
    
    def clear_kb_cache(self) -> None:
        """KB 캐시 삭제"""
        paths = self.get_kb_cache_paths()
        deleted = 0
        for name, path in paths.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted += 1
                    log.info(f"  ✅ 삭제: {name}")
                except Exception as e:
                    log.warning(f"  ⚠️ 삭제 실패: {name} → {e}")
        
        if deleted > 0:
            log.info(f"✅ KB 캐시 삭제 완료: {deleted}개 파일")
        else:
            log.info("ℹ️ 삭제할 KB 캐시 없음")
    
    # ========================================
    # Feedback 코퍼스 캐싱
    # ========================================
    
    def get_fb_cache_paths(self, round_name: str) -> Dict[str, str]:
        """Feedback 코퍼스 캐시 파일 경로 반환"""
        return {
            "scaler": os.path.join(self.fb_cache_dir, f"{round_name}_fb_scaler.pkl"),
            "vectors": os.path.join(self.fb_cache_dir, f"{round_name}_fb_vectors.npy"),
            "annoy": os.path.join(self.fb_cache_dir, f"{round_name}_fb_annoy_index.ann"),
            "metadata": os.path.join(self.fb_cache_dir, f"{round_name}_fb_metadata.pkl"),
            "info": os.path.join(self.fb_cache_dir, f"{round_name}_fb_cache_info.json"),
        }
    
    def compute_fb_hash(self, feedback_corpus: pd.DataFrame) -> str:
        """Feedback 코퍼스 해시 계산"""
        hash_input = f"{len(feedback_corpus)}"
        
        if "case_id" in feedback_corpus.columns:
            case_ids = feedback_corpus["case_id"].astype(str).tolist()
            hash_input += "_" + "_".join(sorted(case_ids[:20]))  # 첫 20개만
        
        if "feedback_label" in feedback_corpus.columns:
            label_dist = feedback_corpus["feedback_label"].value_counts().to_dict()
            hash_input += "_" + str(sorted(label_dist.items()))
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def is_fb_cache_valid(
        self,
        round_name: str,
        feedback_corpus: pd.DataFrame,
        common_features: list,
        alpha: float,
        beta: float,
        gamma: float,
        n_trees: int,
    ) -> bool:
        """Feedback 캐시 유효성 검증"""
        paths = self.get_fb_cache_paths(round_name)
        
        # 1. 파일 존재 여부
        if not all(os.path.exists(p) for p in paths.values()):
            log.info(f"❌ [{round_name}] Feedback 캐시 파일 없음")
            return False
        
        # 2. 캐시 정보 로드
        try:
            with open(paths["info"], "r", encoding="utf-8") as f:
                cache_info = json.load(f)
        except Exception as e:
            log.warning(f"❌ [{round_name}] Feedback 캐시 정보 로드 실패: {e}")
            return False
        
        # 3. 코퍼스 크기 일치
        if cache_info.get("corpus_size") != len(feedback_corpus):
            log.info(f"❌ [{round_name}] Feedback 코퍼스 크기 불일치")
            return False
        
        # 4. 피처 수 일치
        if cache_info.get("n_features") != len(common_features):
            log.info(f"❌ [{round_name}] 피처 수 불일치")
            return False
        
        # 5. 파라미터 일치
        if (cache_info.get("alpha") != alpha or
            cache_info.get("beta") != beta or
            cache_info.get("gamma") != gamma or
            cache_info.get("n_trees") != n_trees):
            log.info(f"❌ [{round_name}] 파라미터 불일치")
            return False
        
        # 6. 데이터 해시 일치
        current_hash = self.compute_fb_hash(feedback_corpus)
        if cache_info.get("corpus_hash") != current_hash:
            log.info(f"❌ [{round_name}] Feedback 코퍼스 해시 불일치")
            return False
        
        log.info(f"✅ [{round_name}] Feedback 캐시 유효성 검증 통과")
        return True
    
    def load_fb_cache(
        self,
        round_name: str,
        n_features: int,
    ) -> Tuple[object, np.ndarray, AnnoyIndex, Dict[str, Any]]:
        """
        Feedback 캐시 로드
        Returns: (scaler, fb_vectors_normalized, annoy_index, metadata)
        """
        log.info(f"📂 [{round_name}] Feedback 캐시 로드 중...")
        paths = self.get_fb_cache_paths(round_name)
        
        # 1. Scaler 로드
        with open(paths["scaler"], "rb") as f:
            scaler = pickle.load(f)
        
        # 2. 벡터 로드
        fb_vectors_normalized = np.load(paths["vectors"])
        
        # 3. Annoy 인덱스 로드
        if not HAS_ANNOY:
            raise ImportError("Annoy가 설치되지 않았습니다: pip install annoy")
        
        annoy_index = AnnoyIndex(n_features, 'angular')
        annoy_index.load(paths["annoy"])
        
        # 4. 메타데이터 로드
        with open(paths["metadata"], "rb") as f:
            metadata = pickle.load(f)
        
        log.info(f"✅ [{round_name}] Feedback 캐시 로드 완료")
        return scaler, fb_vectors_normalized, annoy_index, metadata
    
    def save_fb_cache(
        self,
        round_name: str,
        scaler: object,
        fb_vectors_normalized: np.ndarray,
        annoy_index: AnnoyIndex,
        metadata: Dict[str, Any],
        feedback_corpus: pd.DataFrame,
        common_features: list,
        alpha: float,
        beta: float,
        gamma: float,
        n_trees: int,
    ) -> None:
        """Feedback 캐시 저장"""
        log.info(f"💾 [{round_name}] Feedback 캐시 저장 중...")
        ensure_dir(self.fb_cache_dir)
        paths = self.get_fb_cache_paths(round_name)
        
        # 1. Scaler 저장
        with open(paths["scaler"], "wb") as f:
            pickle.dump(scaler, f)
        
        # 2. 벡터 저장
        np.save(paths["vectors"], fb_vectors_normalized)
        
        # 3. Annoy 인덱스 저장
        annoy_index.save(paths["annoy"])
        
        # 4. 메타데이터 저장
        with open(paths["metadata"], "wb") as f:
            pickle.dump(metadata, f)
        
        # 5. 캐시 정보 저장
        cache_info = {
            "round_name": round_name,
            "corpus_size": len(feedback_corpus),
            "n_features": len(common_features),
            "corpus_hash": self.compute_fb_hash(feedback_corpus),
            "created_at": datetime.now().isoformat(),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n_trees": n_trees,
            "cache_version": "1.0",
        }
        with open(paths["info"], "w", encoding="utf-8") as f:
            json.dump(cache_info, f, indent=2, ensure_ascii=False)
        
        log.info(f"✅ [{round_name}] Feedback 캐시 저장 완료")
    
    def clear_fb_cache(self, round_name: Optional[str] = None) -> None:
        """Feedback 캐시 삭제 (round_name 없으면 전체)"""
        if round_name:
            # 특정 라운드만 삭제
            paths = self.get_fb_cache_paths(round_name)
            deleted = 0
            for name, path in paths.items():
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted += 1
                    except Exception as e:
                        log.warning(f"  ⚠️ 삭제 실패: {name} → {e}")
            
            if deleted > 0:
                log.info(f"✅ [{round_name}] Feedback 캐시 삭제 완료: {deleted}개 파일")
            else:
                log.info(f"ℹ️ [{round_name}] 삭제할 Feedback 캐시 없음")
        else:
            # 전체 삭제
            if os.path.exists(self.fb_cache_dir):
                import shutil
                try:
                    shutil.rmtree(self.fb_cache_dir)
                    log.info("✅ 전체 Feedback 캐시 삭제 완료")
                except Exception as e:
                    log.warning(f"⚠️ Feedback 캐시 삭제 실패: {e}")
            else:
                log.info("ℹ️ 삭제할 Feedback 캐시 없음")
    
    def clear_all_cache(self) -> None:
        """모든 캐시 삭제"""
        log.info("🗑️ 전체 캐시 삭제 중...")
        self.clear_kb_cache()
        self.clear_fb_cache()
        log.info("✅ 전체 캐시 삭제 완료")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 상태 정보 반환"""
        info = {
            "kb_cache": {},
            "fb_cache": {},
        }
        
        # KB 캐시 정보
        kb_info_path = self.get_kb_cache_paths()["info"]
        if os.path.exists(kb_info_path):
            try:
                with open(kb_info_path, "r", encoding="utf-8") as f:
                    info["kb_cache"] = json.load(f)
            except Exception:
                pass
        
        # Feedback 캐시 정보
        if os.path.exists(self.fb_cache_dir):
            fb_files = [f for f in os.listdir(self.fb_cache_dir) if f.endswith("_fb_cache_info.json")]
            for fb_file in fb_files:
                try:
                    with open(os.path.join(self.fb_cache_dir, fb_file), "r", encoding="utf-8") as f:
                        fb_info = json.load(f)
                        round_name = fb_info.get("round_name", "unknown")
                        info["fb_cache"][round_name] = fb_info
                except Exception:
                    pass
        
        return info