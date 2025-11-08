# tools/prioritizer.py
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

def _statistical_flags(flow_count: int, unique_ports: int) -> Tuple[int, int]:
    dos_flag = int(flow_count >= 50 and unique_ports < 2)
    portscan_flag = int(unique_ports >= 50 and flow_count >= 50)
    return dos_flag, portscan_flag

def _statistical_score(flow_count: int, unique_ports: int) -> float:
    dos_flag, pscan_flag = _statistical_flags(flow_count, unique_ports)
    return (dos_flag + pscan_flag) / 2.0  # 0.0, 0.5, 1.0

def _pair_window_stats_directional_from_full_df(full_df: pd.DataFrame) -> pd.DataFrame:
    need = ["capsule_duration", "A_ip", "B_ip", "B_port"]
    df = full_df.copy()
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    grp = df.groupby(["capsule_duration", "A_ip", "B_ip"], dropna=False)
    stats = grp.agg(
        flow_count=("B_port", "size"),
        unique_ports=("B_port", lambda s: s.nunique(dropna=False)),
    ).reset_index()
    return stats

class PrioritizerTool:
    """
    Front-door Alert Refinement (게이팅) 어댑터.
    - 입력: predictions CSV(DataFrame) 또는 Detection 단계에서 만든 Alerts DataFrame
    - 출력: 확신도 낮은 케이스 하위 X%(또는 Top-K) DataFrame
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 bottom_percent: Optional[float] = 5.0, top_k: Optional[int] = None,
                 out_dir: Optional[str] = "./feedback_cases"):
        if bottom_percent is not None and top_k is not None:
            raise ValueError("bottom_percent 또는 top_k 중 하나만 설정하세요.")
        self.alpha = alpha
        self.beta = beta
        self.bottom_percent = bottom_percent
        self.top_k = top_k
        self.out_dir = out_dir

    def select(self, df_with_predictions: pd.DataFrame, round_name: str) -> pd.DataFrame:
        req = ["predicted_label","attack_probability","prediction_confidence",
               "label","normal_probability","is_correct"]
        miss = [c for c in req if c not in df_with_predictions.columns]
        if miss:
            raise ValueError(f"필수 컬럼 누락: {miss}")

        # Attack만 추출
        att = df_with_predictions[df_with_predictions["predicted_label"] == "Attack"].copy()
        if att.empty:
            return pd.DataFrame()

        # 방향 고려 통계 merge
        stats_all = _pair_window_stats_directional_from_full_df(df_with_predictions)
        att = att.merge(stats_all, on=["capsule_duration","A_ip","B_ip"], how="left", validate="m:1")
        att["flow_count"]   = pd.to_numeric(att["flow_count"], errors="coerce").fillna(0).astype(int)
        att["unique_ports"] = pd.to_numeric(att["unique_ports"], errors="coerce").fillna(0).astype(int)

        # confidence_score = α*ap + β*stat
        att["statistical_score"]  = att.apply(lambda r: _statistical_score(int(r["flow_count"]), int(r["unique_ports"])), axis=1)
        att["attack_probability"] = pd.to_numeric(att["attack_probability"], errors="coerce").fillna(0.0)
        att["confidence_score"]   = self.alpha * att["attack_probability"] + self.beta * att["statistical_score"]

        att = att.sort_values(["confidence_score","attack_probability"], ascending=[True, True])

        # 하위 X% or Top-K
        if self.bottom_percent is not None:
            k = max(1, int(len(att) * (self.bottom_percent/100.0)))
        else:
            k = min(self.top_k, len(att))
        targets = att.head(k).copy()

        # case_id 부여 및 출력 경로(선택)
        targets = targets.reset_index(drop=True)
        targets.insert(0, "case_id", [f"{round_name}_lowconf_{i+1:04d}" for i in range(len(targets))])

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f"{round_name}_low_confidence_cases.csv")
            targets.to_csv(out_path, index=False, encoding="utf-8")

        return targets
