# -*- coding: utf-8 -*-
"""
Knowledge Base Manager
Train 시점의 labeled 사례들을 로드하고 관리하는 모듈
"""
from __future__ import annotations
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from tools.base import get_logger

log = get_logger("KnowledgeBase")


class KnowledgeBase:
    """
    Train_Cases 디렉토리에서 labeled 사례를 로드하여 관리
    """
    def __init__(self, train_cases_dir: str = "/Train_Cases"):
        self.train_cases_dir = train_cases_dir
        self.kb_df = None
        self.is_loaded = False

    def load(self) -> bool:
        """
        Train_Cases 디렉토리에서 모든 CSV 파일 로드
        Returns: 성공 여부
        """
        if not os.path.exists(self.train_cases_dir):
            log.warning(f"Train_Cases 디렉토리 없음: {self.train_cases_dir}")
            return False

        files = sorted(glob.glob(os.path.join(self.train_cases_dir, "*.csv")))
        if not files:
            log.warning(f"Train_Cases에 CSV 파일 없음: {self.train_cases_dir}")
            return False

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                if "label" in df.columns and len(df) > 0:
                    dfs.append(df)
                    log.info(f"로드: {os.path.basename(f)} ({len(df)} rows)")
            except Exception as e:
                log.warning(f"로드 실패: {f} → {e}")
                continue

        if not dfs:
            log.warning("로드된 labeled 사례 없음")
            return False

        self.kb_df = pd.concat(dfs, ignore_index=True)
        
        # 필수 컬럼 확인
        required = ["label"]
        missing = [c for c in required if c not in self.kb_df.columns]
        if missing:
            log.error(f"필수 컬럼 누락: {missing}")
            return False

        self.is_loaded = True
        log.info(f"Knowledge Base 로드 완료: {len(self.kb_df)} rows")
        return True

    def get_labeled_cases(self, labels: List[str] = None) -> pd.DataFrame:
        """
        특정 라벨의 사례만 반환
        Args:
            labels: 필터링할 라벨 리스트 (None이면 모든 라벨)
        Returns:
            필터링된 DataFrame
        """
        if not self.is_loaded or self.kb_df is None:
            return pd.DataFrame()

        if labels is None:
            return self.kb_df.copy()

        return self.kb_df[self.kb_df["label"].isin(labels)].copy()

    def get_stats(self) -> Dict[str, int]:
        """Knowledge Base 통계 반환"""
        if not self.is_loaded or self.kb_df is None:
            return {}

        return {
            "total": len(self.kb_df),
            "labels": self.kb_df["label"].value_counts().to_dict() if "label" in self.kb_df.columns else {},
            "files_loaded": len(glob.glob(os.path.join(self.train_cases_dir, "*.csv")))
        }

    def export_as_feedback_corpus(self, out_dir: str = "./feedback_cases") -> str:
        """
        Knowledge Base를 피드백 코퍼스 형식으로 Export
        Returns: 출력 파일 경로
        """
        if not self.is_loaded or self.kb_df is None:
            log.error("Knowledge Base 미로드")
            return ""

        os.makedirs(out_dir, exist_ok=True)
        
        # 피드백 형식으로 변환
        export_df = self.kb_df.copy()
        
        # 필요한 피드백 컬럼 추가
        for col, default in [
            ("case_id", ""),
            ("feedback_label", ""),
            ("feedback_confidence", ""),
            ("feedback_reason", ""),
            ("reviewed", True),
            ("needs_review", False),
            ("review_date", ""),
        ]:
            if col not in export_df.columns:
                if col == "reviewed":
                    export_df[col] = True
                elif col == "case_id" and "case_id" not in export_df.columns:
                    export_df[col] = [f"KB_{i:06d}" for i in range(len(export_df))]
                elif col == "feedback_label":
                    export_df[col] = export_df.get("label", "")
                elif col == "feedback_confidence":
                    export_df[col] = 5  # 최고 신뢰도
                elif col == "feedback_reason":
                    export_df[col] = "(Train Knowledge Base)"
                else:
                    export_df[col] = default

        out_path = os.path.join(out_dir, "Knowledge_Base_Corpus.csv")
        export_df.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"Knowledge Base Export: {out_path} ({len(export_df)} rows)")
        return out_path

    def __len__(self) -> int:
        return len(self.kb_df) if self.is_loaded and self.kb_df is not None else 0

    def __repr__(self) -> str:
        if not self.is_loaded:
            return "KnowledgeBase(not loaded)"
        stats = self.get_stats()
        return f"KnowledgeBase(total={stats.get('total', 0)}, labels={stats.get('labels', {})})"