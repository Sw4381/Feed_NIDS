# -*- coding: utf-8 -*-
"""
DecisionPolicy
- RAG 검색 결과를 기반으로 Alert의 Label을 재조정
- 유사 사례의 Label을 참조하여 자동으로 Label 결정
"""
from typing import List, Dict, Any, Optional

class DecisionPolicy:
    """
    유사 사례 기반 Label 재조정 정책
    """
    def __init__(self,
                 alpha_ip: float = 0.5,
                 beta_cos: float = 0.3,
                 gamma_overlap: float = 0.2,
                 threshold: float = 0.7,
                 top_refs: int = 3,
                 label_confidence_threshold: float = 0.6):
        """
        Args:
            alpha_ip: IP 일치 가중치
            beta_cos: 코사인 유사도 가중치
            gamma_overlap: 특징 중복 가중치
            threshold: 자동 결정 임계값
            top_refs: 상위 참조 케이스 수
            label_confidence_threshold: Label 재조정을 위한 최소 유사도 (0.6 = 60%)
        """
        self.alpha_ip = alpha_ip
        self.beta_cos = beta_cos
        self.gamma_overlap = gamma_overlap
        self.threshold = threshold
        self.top_refs = top_refs
        self.label_confidence_threshold = label_confidence_threshold

    def decide(self, alert_id: str, candidates: List[Dict[str, Any]], 
               original_label: str = "Attack") -> Dict[str, Any]:
        """
        유사 사례를 기반으로 Label 재조정 및 결정
        
        Args:
            alert_id: Alert ID
            candidates: RAG 검색 결과 [{"ref_id", "score", "label", "src", "dst"}, ...]
            original_label: 원래 예측된 Label (기본값: "Attack")
            
        Returns:
            {
                "alert_id": str,
                "decision": str,  # 의사결정 타입
                "adjusted_label": str,  # 재조정된 Label
                "confidence": float,  # 재조정 확신도
                "reference_cases": List[Dict],  # 참조한 유사 사례들
                "reasoning": str  # 재조정 근거
            }
        """
        if not candidates:
            return {
                "alert_id": alert_id,
                "decision": "NO_SIMILAR_CASE",
                "adjusted_label": original_label,  # 원래 Label 유지
                "confidence": 0.0,
                "reference_cases": [],
                "reasoning": "유사 사례가 없어 원래 예측 Label 유지"
            }

        # Top-K 유사 사례 선택
        top_cases = candidates[:self.top_refs]
        
        # Label이 있는 케이스만 필터링
        labeled_cases = [c for c in top_cases if c.get("label") and c.get("label") != ""]
        
        if not labeled_cases:
            return {
                "alert_id": alert_id,
                "decision": "NO_LABELED_CASE",
                "adjusted_label": original_label,
                "confidence": 0.0,
                "reference_cases": top_cases,
                "reasoning": "유사 사례는 있지만 Label이 없어 원래 예측 유지"
            }

        # 가장 유사도가 높은 케이스 선택
        best_case = labeled_cases[0]
        best_similarity = best_case.get("score", 0.0)
        best_label = best_case.get("label", "")

        # Label 분포 계산 (투표 방식)
        label_votes = {}
        weighted_votes = {}
        
        for case in labeled_cases:
            lbl = case.get("label", "")
            sim = case.get("score", 0.0)
            
            if lbl:
                label_votes[lbl] = label_votes.get(lbl, 0) + 1
                weighted_votes[lbl] = weighted_votes.get(lbl, 0.0) + sim

        # 가중 투표로 최종 Label 결정
        if weighted_votes:
            adjusted_label = max(weighted_votes.items(), key=lambda x: x[1])[0]
            vote_count = label_votes.get(adjusted_label, 0)
            avg_similarity = weighted_votes[adjusted_label] / vote_count
        else:
            adjusted_label = original_label
            avg_similarity = 0.0

        # 확신도 기반 의사결정
        if best_similarity >= self.label_confidence_threshold:
            # 고신뢰 재조정
            if adjusted_label == "Normal":
                decision = "AUTO_ADJUST_TO_NORMAL"
                reasoning = f"유사도 {best_similarity:.2f}로 {vote_count}개 사례에서 Normal로 재조정"
            elif adjusted_label == "Attack":
                decision = "AUTO_CONFIRM_ATTACK"
                reasoning = f"유사도 {best_similarity:.2f}로 {vote_count}개 사례에서 Attack 확인"
            else:
                decision = "AUTO_ADJUST_TO_OTHER"
                reasoning = f"유사도 {best_similarity:.2f}로 {adjusted_label}로 재조정"
            
            confidence = avg_similarity
            
        elif best_similarity >= self.threshold:
            # 중신뢰 - 참고만
            decision = "REFER_TO_SIMILAR"
            adjusted_label = original_label  # 원래 Label 유지하되 참고
            confidence = avg_similarity
            reasoning = f"유사도 {best_similarity:.2f}로 참고 가능하나 원래 예측 유지"
            
        else:
            # 저신뢰 - 유사도 부족
            decision = "LOW_SIMILARITY"
            adjusted_label = original_label
            confidence = avg_similarity
            reasoning = f"유사도 {best_similarity:.2f}로 낮아 원래 예측 유지"

        return {
            "alert_id": alert_id,
            "decision": decision,
            "adjusted_label": adjusted_label,
            "original_label": original_label,
            "confidence": float(confidence),
            "reference_cases": [
                {
                    "ref_id": c.get("ref_id", ""),
                    "label": c.get("label", ""),
                    "similarity": c.get("score", 0.0),
                    "src": c.get("src", ""),
                    "dst": c.get("dst", "")
                }
                for c in labeled_cases
            ],
            "reasoning": reasoning,
            "label_distribution": label_votes,
            "best_similarity": float(best_similarity)
        }