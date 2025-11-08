# tools/evidence_pack.py
import shap
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

def _calc_shap_values(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        if hasattr(model, "get_booster") or hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):  # binary → class 1
            vals = vals[1]
        return vals
    except Exception as e:
        print(f"[EvidencePack] SHAP 계산 실패: {e}")
        return None

def _topk_from_shap(shap_vals: np.ndarray, feature_cols: List[str], row_idx: int, k: int = 5):
    abs_row = np.abs(shap_vals[row_idx])
    idx = np.argsort(abs_row)[-k:][::-1]
    top = [{"name": feature_cols[j], "value": float(abs_row[j])} for j in idx]
    return top, idx

class EvidencePackTool:
    """
    선별된 케이스 DataFrame을 Evidence Pack 리스트로 변환.
    - SHAP Top-K (이름/SHAP값) + 원시값(raw feature values) 포함
    """
    def __init__(self, model, feature_columns: List[str], shap_topk: int = 5):
        self.model = model
        self.feature_columns = feature_columns
        self.shap_topk = shap_topk

    def build_from_targets(self, targets_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if targets_df.empty:
            return []
        # 사용 가능한 피처만 추림
        avail = [c for c in self.feature_columns if c in targets_df.columns]
        X = targets_df[avail].copy()

        shap_vals = _calc_shap_values(self.model, X) if self.model is not None else None

        packs: List[Dict[str, Any]] = []
        for i, row in targets_df.reset_index(drop=True).iterrows():
            topk, idxs = ([], [])
            if shap_vals is not None:
                topk, idxs = _topk_from_shap(shap_vals, avail, i, k=self.shap_topk)

            rawvals = []
            for j in idxs if len(idxs) else range(min(self.shap_topk, len(avail))):
                name = avail[j]
                rawvals.append({"name": name, "value": float(row[name])})

            pack = {
                "alert_id": row.get("alert_id", row.get("case_id", f"case_{i}")),
                "ip_pair": {
                    "src": row.get("A_ip", ""),
                    "dst": row.get("B_ip", ""),
                    "proto": str(row.get("Protocol", row.get("proto",""))),
                    "sport": int(row.get("A_port", row.get("src_port", 0))),
                    "dport": int(row.get("B_port", row.get("dst_port", 0))),
                },
                "topk_shap": topk,          # [{"name": "...","value": ...}, ...]
                "topk_raw": rawvals,        # [{"name": "...","value": ...}, ...]
                "meta": {
                    "round": row.get("round_name", ""),
                    "capsule_duration": row.get("capsule_duration",""),
                    "flow_count": int(row.get("flow_count", 0)),
                    "unique_ports": int(row.get("unique_ports", 0)),
                    "attack_probability": float(row.get("attack_probability", 0.0)),
                    "statistical_score": float(row.get("statistical_score", 0.0)),
                    "confidence_score": float(row.get("confidence_score", 0.0)),
                }
            }
            packs.append(pack)
        return packs
