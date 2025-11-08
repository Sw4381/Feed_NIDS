# -*- coding: utf-8 -*-
from typing import List, Dict, Any

class DecisionPolicy:
    """
    간단한 결정 규칙:
      - 최상위 후보 score >= threshold:
         · 후보 label == "Normal" → AUTO_SUPPRESS (정상화)
         · 후보 label == "Attack" → AUTO_CONFIRM_ATTACK
      - 그 외 → ROUTE_ANALYST
    """
    def __init__(self, alpha_ip: float = 0.5, beta_cos: float = 0.3, gamma_overlap: float = 0.2,
                 threshold: float = 0.7, top_refs: int = 3):
        self.alpha_ip = float(alpha_ip)
        self.beta_cos = float(beta_cos)
        self.gamma_overlap = float(gamma_overlap)
        self.threshold = float(threshold)
        self.top_refs = int(top_refs)

    def decide(self, alert_id: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return {"alert_id": alert_id, "decision": "ROUTE_ANALYST", "reason": "No candidates"}

        best = max(candidates, key=lambda c: c.get("score", 0.0))
        score = float(best.get("score", 0.0))
        label = (best.get("label") or "").strip()

        if score >= self.threshold and label:
            if label == "Normal":
                return {
                    "alert_id": alert_id,
                    "decision": "AUTO_SUPPRESS",
                    "reason": "Matched Normal evidence",
                    "score": score,
                    "ref_id": best.get("ref_id", "")
                }
            if label == "Attack":
                return {
                    "alert_id": alert_id,
                    "decision": "AUTO_CONFIRM_ATTACK",
                    "reason": "Matched Attack evidence",
                    "score": score,
                    "ref_id": best.get("ref_id", "")
                }

        return {"alert_id": alert_id, "decision": "ROUTE_ANALYST",
                "reason": "Low similarity or no label", "score": score}
