# tools/detection.py
import os
import glob
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

class DetectionTool:
    """
    FEED-NIDS Detection 어댑터.
    - 모델을 로드하고, 라운드 CSV 또는 DataFrame에 대해 예측을 수행
    - 오케스트레이터가 소비할 수 있는 Alert 리스트를 반환
    """
    def __init__(
        self,
        model_path: str,
        feature_order: Optional[List[str]] = None,
        rounds_directory: Optional[str] = None,
        label_col: str = "label"
    ):
        self.model_path = model_path
        self.feature_order = feature_order  # 명시 제공 시 그대로 사용
        self.rounds_directory = rounds_directory
        self.label_col = label_col
        self.model = None
        self._load_model_and_features()

    # ---------- 내부 유틸 ----------
    def _load_model_and_features(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

        # feature 순서 자동 감지
        if self.feature_order is None:
            if hasattr(self.model, "feature_names_in_"):
                self.feature_order = list(self.model.feature_names_in_)
            elif hasattr(self.model, "get_booster") and hasattr(self.model.get_booster(), "feature_names"):
                self.feature_order = list(self.model.get_booster().feature_names)
            else:
                # rounds_directory에서 첫 파일로 추론
                if not self.rounds_directory:
                    raise ValueError("feature_order를 알 수 없습니다. rounds_directory를 지정하거나 feature_order를 넘겨주세요.")
                round_files = sorted(glob.glob(os.path.join(self.rounds_directory, "Round_*.csv")))
                if not round_files:
                    raise FileNotFoundError(f"No round files found in {self.rounds_directory}")
                first = pd.read_csv(round_files[0], nrows=1)
                self.feature_order = [c for c in first.columns if c != self.label_col]

    def _prepare_X(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features in df: {missing}")
        X = df[self.feature_order]
        y = None
        if self.label_col in df.columns:
            # 0/1로 맵핑 (없어도 동작 가능)
            y = df[self.label_col].map({"Normal": 0, "Attack": 1}).astype("float").fillna(-1).astype(int)
        return X, y

    # ---------- 공개 메서드 ----------
    def run_on_round_file(self, round_csv_path: str) -> List[Dict]:
        """라운드 CSV 파일 단위로 예측 → Alert 리스트 반환"""
        round_name = os.path.basename(round_csv_path).replace(".csv", "")
        df = pd.read_csv(round_csv_path, low_memory=False)
        return self.run_on_dataframe(df, round_name=round_name)

    def run_on_dataframe(self, df: pd.DataFrame, round_name: str = "Round_UNKNOWN") -> List[Dict]:
        """
        DataFrame 단위로 예측 수행.
        반환: 오케스트레이터 Alert 스키마에 맞춘 dict 리스트
        """
        X, _ = self._prepare_X(df)
        # 예측
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # 결과 합성
        alerts: List[Dict] = []
        for i in range(len(df)):
            src_ip = df.iloc[i].get("src_ip", None) or df.iloc[i].get("Src IP", None) or ""
            dst_ip = df.iloc[i].get("dst_ip", None) or df.iloc[i].get("Dst IP", None) or ""
            proto  = str(df.iloc[i].get("proto", df.iloc[i].get("Protocol", "")))
            sport  = int(df.iloc[i].get("src_port", df.iloc[i].get("Src Port", 0)))
            dport  = int(df.iloc[i].get("dst_port", df.iloc[i].get("Dst Port", 0)))

            ai_score_attack = float(y_proba[i, 1])      # Attack 확률
            ai_score_normal = float(y_proba[i, 0])
            pred_label = "Attack" if int(y_pred[i]) == 1 else "Normal"
            confidence = float(max(ai_score_attack, ai_score_normal))

            alert = {
                "alert_id": f"{round_name}-{i}",
                "round_name": round_name,
                "time_window": df.iloc[i].get("time_window", None) or "",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": sport,
                "dst_port": dport,
                "proto": proto,
                "ai_score": ai_score_attack,                 # 오케스트레이터의 우선순위 계산에 사용
                "prediction_confidence": confidence,
                "predicted_label": pred_label,
                "raw_features": df.iloc[i][self.feature_order].to_dict()
            }
            # 원 라벨이 있으면 첨부(평가/로그용)
            if self.label_col in df.columns:
                alert["true_label"] = df.iloc[i][self.label_col]
            alerts.append(alert)

        return alerts

    def feature_names(self) -> List[str]:
        return list(self.feature_order)
