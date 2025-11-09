#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEED-NIDS Pipeline Orchestrator v3 (캐싱 통합)
실행 흐름:
  Detection → Prioritizer → (선택) Train KB → (선택) Feedback → 최종 결과
  
새 기능:
  - Feedback 코퍼스 캐싱
  - --force-rebuild-cache 플래그
  - --cache-dir 경로 설정
"""

import sys
if sys.version_info < (3, 8):
    raise SystemExit("Python 3.8+ 필요합니다.")

import os
import glob
import argparse
import pandas as pd
import numpy as np

from tools.base import get_logger, ToolResult
from tools.detection import DetectionTool
from tools.prioritizer import PrioritizerTool
from tools.auto_feedback import AutoFeedbackTool
from tools.merge import MergeTool
from tools.knowledge_base import KnowledgeBase

# 최적화 버전 임포트 시도
try:
    from tools.similarity_apply import (
        OptimizedSimilarityApplyTool,
        OptimizedKBSimilarityApplyTool,
        HAS_ANNOY,
    )
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    HAS_ANNOY = False

# 캐시 매니저 임포트
try:
    from cache_manager import CacheManager
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

log = get_logger("Orchestrator-v3")


def list_rounds_from_predictions(pred_dir: str):
    """round_predictions에서 Round 목록 산출"""
    paths = sorted(glob.glob(os.path.join(pred_dir, "*_with_predictions.csv")))
    return [os.path.basename(p).replace("_with_predictions.csv", "") for p in paths]


def list_rounds_from_inputs(det_in_dir: str):
    """입력 폴더에서 Round 목록 산출"""
    names = set()
    for p in glob.glob(os.path.join(det_in_dir, "Round_*.csv")):
        bn = os.path.basename(p)
        rn = bn[:-4]
        if rn.endswith("_raw"):
            rn = rn[:-4]
        if rn.startswith("Round_"):
            names.add(rn)
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(
        description="FEED-NIDS v3: Detection → Prioritizer → (Train KB) → (Feedback) [캐싱 지원]"
    )
    
    # 실행 모드
    ap.add_argument(
        "--mode", 
        choices=["kb-only", "feedback-only", "full"],
        default="full",
        help="kb-only: Train KB만, feedback-only: Feedback만, full: 전체(권장)"
    )
    
    # 라운드 선택
    ap.add_argument("--rounds", nargs="*", default=None, help="예: Round_1 Round_3")
    ap.add_argument("--all", action="store_true", help="모든 라운드 처리")
    
    # 디렉토리 경로
    ap.add_argument("--det-in", default="./test_rounds", help="Detection 입력")
    ap.add_argument("--pred-dir", default="./round_predictions", help="예측 결과")
    ap.add_argument("--feedback-dir", default="./feedback_cases", help="피드백 케이스")
    ap.add_argument("--applied-dir", default="./round_predictions_applied", help="적용 결과")
    ap.add_argument("--train-cases-dir", default="./Train_Cases", help="Train KB 위치")
    ap.add_argument("--model-path", default="./models/xgboost_binary_classifier.joblib")
    ap.add_argument("--det-out", default="./round_results")
    ap.add_argument("--cache-dir", default="./cache", help="캐시 저장 디렉토리")

    # Detection (0단계)
    ap.add_argument("--skip-detection", action="store_true", help="Detection 스킵")
    ap.add_argument("--force-detection", action="store_true", help="Detection 재실행")
    ap.add_argument("--det-threshold", type=float, default=0.5)

    # 최적화 및 캐싱
    ap.add_argument("--use-optimized-search", action="store_true", help="최적화 검색 활성화 (Annoy)")
    ap.add_argument("--annoy-n-trees", type=int, default=100, help="Annoy 트리 개수")
    ap.add_argument("--annoy-top-k", type=int, default=1000, help="Annoy 검색 Top-K")
    ap.add_argument("--kb-annoy-top-k", type=int, default=1000, help="KB Annoy 검색 Top-K")
    ap.add_argument("--force-rebuild-cache", action="store_true", help="캐시 강제 재빌드")
    ap.add_argument("--no-cache", action="store_true", help="캐싱 비활성화")

    # Phase 1: Train KB 파라미터
    ap.add_argument("--kb-alpha", type=float, default=0.3, help="KB IP 가중치")
    ap.add_argument("--kb-beta", type=float, default=0.4, help="KB Cosine 가중치")
    ap.add_argument("--kb-gamma", type=float, default=0.3, help="KB SHAP 가중치")
    ap.add_argument("--kb-threshold", type=float, default=0.6, help="KB 유사도 임계값")
    ap.add_argument("--kb-no-direction", action="store_true", help="KB 방향 무시")
    ap.add_argument("--kb-top-k", type=int, default=5, help="KB SHAP Top-K")

    # Phase 2: Gating 파라미터
    ap.add_argument("--gate-alpha", type=float, default=0.3, help="Gating: 공격 확률")
    ap.add_argument("--gate-beta", type=float, default=0.7, help="Gating: 통계 점수")
    ap.add_argument("--gate-bottom-percent", type=float, default=5.0, help="Gating: 하위%")
    ap.add_argument("--gate-top-k", type=int, default=None, help="Gating: 고정 K")
    ap.add_argument("--gate-no-shap", action="store_true", help="Gating: SHAP 비활성화")

    # Phase 2: Auto-feedback
    ap.add_argument("--skip-auto-feedback", action="store_true")
    ap.add_argument("--auto-top-n", type=int, default=50)
    ap.add_argument("--auto-percent", type=float, default=None)

    # Phase 2: Similarity Apply (Feedback)
    ap.add_argument("--alpha", type=float, default=0.3, help="Feedback IP 가중치")
    ap.add_argument("--beta", type=float, default=0.4, help="Feedback Cosine 가중치")
    ap.add_argument("--gamma", type=float, default=0.3, help="Feedback SHAP 가중치")
    ap.add_argument("--threshold", type=float, default=0.6, help="Feedback 임계값")
    ap.add_argument("--no-direction", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)

    # 라운드별 스킵
    ap.add_argument("--skip-feedback-rounds", nargs="*", default=[])
    ap.add_argument("--skip-feedback-round1", action="store_true")

    args = ap.parse_args()

    # ===== 라운드 결정 =====
    if args.rounds:
        rounds = args.rounds
    elif args.all:
        rounds = list_rounds_from_predictions(args.pred_dir)
        if not rounds:
            rounds = list_rounds_from_inputs(args.det_in)
    else:
        log.error("처리할 라운드 없음. (--rounds ... 또는 --all)")
        return

    if not rounds:
        log.error("라운드 목록 없음")
        return

    # ===== 캐시 매니저 초기화 =====
    cache_manager = None
    use_cache = not args.no_cache and HAS_CACHE
    
    if args.force_rebuild_cache and use_cache:
        log.info("🗑️ 기존 캐시 삭제 (--force-rebuild-cache)")
        cache_manager = CacheManager(cache_root=args.cache_dir)
        cache_manager.clear_all_cache()
    
    if use_cache:
        cache_manager = CacheManager(cache_root=args.cache_dir)

    # ===== 로그 출력 =====
    log.info("=" * 70)
    log.info(f"처리 라운드: {rounds}")
    log.info(f"실행 모드: {args.mode} (kb-only/feedback-only/full)")
    
    if args.use_optimized_search and HAS_OPTIMIZED:
        log.info(f"최적화: ✅ 활성화 (Annoy n_trees={args.annoy_n_trees})")
        if use_cache:
            log.info(f"캐싱: ✅ 활성화 (cache_dir={args.cache_dir})")
        else:
            log.info("캐싱: ⚠️ 비활성화")
    else:
        if args.use_optimized_search and not HAS_OPTIMIZED:
            log.warning("최적화: ⚠️ Annoy 미설치 → 기본 버전 사용")
        log.info("최적화: ❌ 비활성화")
    
    log.info("=" * 70)

    # ===== Phase 0: Detection =====
    if not args.skip_detection:
        log.info("Phase 0️⃣: Detection 시작")
        det = DetectionTool(
            rounds_directory=args.det_in,
            predictions_directory=args.pred_dir,
            results_directory=args.det_out,
            model_path=args.model_path,
            rounds=rounds,
            threshold=args.det_threshold,
            force=args.force_detection
        ).run()
        if not det.ok:
            log.error(f"Detection 실패: {det.message}")
            return
        log.info("✅ Phase 0: Detection 완료")
    else:
        log.info("⭐ Phase 0: Detection 스킵")

    log.info("")

    # ===== Phase 1️⃣: Prioritizer (분석 대상 추출) - 항상 실행 =====
    log.info("Phase 1️⃣: Prioritizer (분석 대상 추출)")
    log.info("-" * 70)

    prioritizer_results = {}
    for rn in rounds:
        log.info(f"[{rn}] Prioritizer 시작")
        pr = PrioritizerTool(
            round_name=rn,
            pred_dir=args.pred_dir,
            out_dir=args.feedback_dir,
            alpha=args.gate_alpha,
            beta=args.gate_beta,
            bottom_percent=args.gate_bottom_percent if args.gate_top_k is None else None,
            top_k=args.gate_top_k,
            model_path=args.model_path,
            enable_shap=not args.gate_no_shap,
        ).run()
        
        prioritizer_results[rn] = pr
        if pr.ok:
            log.info(f"✅ [{rn}] Prioritizer 완료: {pr.data}")
        else:
            log.warning(f"⚠️ [{rn}] Prioritizer: {pr.message}")

    log.info("✅ Phase 1: Prioritizer 완료")
    log.info("")

    # ===== Phase 2️⃣: Train Knowledge Base 적용 (선택사항) =====
    if args.mode in ["full", "kb-only"]:
        log.info("Phase 2️⃣: Train Knowledge Base 적용 (선택사항)")
        log.info("-" * 70)

        # KB 로드
        kb = KnowledgeBase(args.train_cases_dir)
        if kb.load():
            stats = kb.get_stats()
            log.info(f"✅ Knowledge Base 로드: {stats}")
            kb_corpus = kb.kb_df.copy()
        else:
            log.warning("⚠️ Knowledge Base 로드 실패 → Phase 2 스킵")
            kb_corpus = None

        # 모든 라운드에 KB 적용
        kb_results = {}
        if kb_corpus is not None:
            for rn in rounds:
                log.info(f"[{rn}] KB 적용 시작")
                
                kb_tool = OptimizedKBSimilarityApplyTool(
                    round_name=rn,
                    pred_dir=args.pred_dir,  # ✅ 수정됨: args.feedback_dir → args.pred_dir
                    kb_corpus=kb_corpus,
                    out_dir=args.pred_dir,
                    alpha=args.kb_alpha,
                    beta=args.kb_beta,
                    gamma=args.kb_gamma,
                    threshold=args.kb_threshold,
                    direction_sensitive=not args.kb_no_direction,
                    top_k=args.kb_top_k,
                    n_trees=args.annoy_n_trees,
                    annoy_search_k=args.kb_annoy_top_k,
                    cache_dir=args.cache_dir,
                    use_cache=use_cache,
                ).run()
                
                kb_results[rn] = kb_tool
                if kb_tool.ok:
                    log.info(f"✅ [{rn}] KB 적용 완료: {kb_tool.data}")
                else:
                    log.warning(f"⚠️ [{rn}] KB: {kb_tool.message}")

        log.info("✅ Phase 2: Train KB 완료")
        log.info("")

        # kb-only 모드면 여기서 종료
        if args.mode == "kb-only":
            log.info("=" * 70)
            log.info("🎉 kb-only 모드 완료!")
            log.info("=" * 70)
            return

    else:
        log.info("⭐ Phase 2: Train KB 스킵 (--mode feedback-only)")
        log.info("")

    # ===== Phase 3️⃣: Feedback 적용 (선택사항) =====
    if args.mode in ["full", "feedback-only"]:
        log.info("Phase 3️⃣: Feedback 적용 (선택사항)")
        log.info("-" * 70)

        if args.skip_feedback_round1:
            args.skip_feedback_rounds = list(set(args.skip_feedback_rounds + ["Round_1"]))
        skip_set = set(args.skip_feedback_rounds)

        for rn in rounds:
            log.info(f"[{rn}] Feedback 처리 시작")

            # Step 1: 유사도 적용 (이전 라운드 피드백 기반 자동 수정)
            if rn in skip_set:
                log.info(f"  └─ SimilarityApply 스킵")
            else:
                log.info(f"  └─ SimilarityApply (이전 피드백 기반 자동 수정)")
                
                log.info(f"    [최적화 Feedback 검색 사용]")
                sa = OptimizedSimilarityApplyTool(
                    round_name=rn,
                    feedback_dir=args.feedback_dir,
                    out_dir=args.applied_dir,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    threshold=args.threshold,
                    direction_sensitive=not args.no_direction,
                    top_k=args.top_k,
                    n_trees=args.annoy_n_trees,
                    annoy_search_k=args.annoy_top_k,
                    cache_dir=args.cache_dir,
                    use_cache=use_cache,
                ).run()
                
                if sa.ok:
                    log.info(f"    ✅ SimilarityApply 완료: {sa.data}")
                else:
                    log.error(f"    ❌ SimilarityApply 실패: {sa.message}")
                    continue

            log.info(f"  └AutoFeedback ")
            af = AutoFeedbackTool(
                round_name=rn,
                feedback_dir=args.feedback_dir,
                top_n=args.auto_top_n,
                percent=args.auto_percent,
            ).run()
            if af.ok:
                log.info(f"    ✅ AutoFeedback 완료")
            else:
                log.warning(f"    ⚠️ AutoFeedback: {af.message}")

            # Step 3: Merge (최종 결과 병합)
            log.info(f"  └─ Merge")
            mg = MergeTool(
                round_name=rn,
                pred_dir=args.pred_dir,
                applied_dir=args.applied_dir,
            ).run()
            
            if mg.ok:
                log.info(f"    ✅ Merge 완료")
            else:
                log.error(f"    ❌ Merge 실패: {mg.message}")

            log.info(f"✅ [{rn}] Feedback 처리 완료")

        log.info("✅ Phase 3: Feedback 완료")
        log.info("")

    else:
        log.info("⭐ Phase 3: Feedback 스킵 (--mode kb-only)")
        log.info("")

    # ===== 캐시 정보 출력 =====
    if use_cache and cache_manager:
        log.info("=" * 70)
        log.info("📦 캐시 정보")
        log.info("-" * 70)
        cache_info = cache_manager.get_cache_info()
        
        if cache_info["kb_cache"]:
            kb_info = cache_info["kb_cache"]
            log.info(f"KB 캐시:")
            log.info(f"  - KB 크기: {kb_info.get('kb_size', 'N/A')}")
            log.info(f"  - 피처 수: {kb_info.get('n_features', 'N/A')}")
            log.info(f"  - 생성 시각: {kb_info.get('created_at', 'N/A')}")
        
        if cache_info["fb_cache"]:
            log.info(f"Feedback 캐시: {len(cache_info['fb_cache'])}개 라운드")
            for rn, fb_info in sorted(cache_info["fb_cache"].items()):
                log.info(f"  - {rn}: {fb_info.get('corpus_size', 'N/A')} 케이스")

    log.info("=" * 70)
    log.info("🎉 전체 파이프라인 완료!")
    log.info("=" * 70)


if __name__ == "__main__":
    main()