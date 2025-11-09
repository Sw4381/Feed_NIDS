#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if sys.version_info < (3, 8):
    raise SystemExit("Python 3.8+ 필요합니다. 예: `python3 pipeline_orchestrator.py ...`")

import os
import glob
import argparse
import pandas as pd
import numpy as np

# tools
from tools.base import get_logger
from tools.prioritizer import PrioritizerTool
from tools.auto_feedback import AutoFeedbackTool
from tools.similarity_apply import SimilarityApplyTool
from tools.merge import MergeTool
from tools.detection import DetectionTool  # ⬅️ 내장 Detection 사용

log = get_logger("Orchestrator")


def list_rounds_from_predictions(pred_dir: str):
    """round_predictions 안의 *_with_predictions.csv 기준으로 Round 목록 산출"""
    paths = sorted(glob.glob(os.path.join(pred_dir, "*_with_predictions.csv")))
    return [os.path.basename(p).replace("_with_predictions.csv", "") for p in paths]


def list_rounds_from_inputs(det_in_dir: str):
    """입력 폴더(det_in)에서 Round_* 입력 CSV를 스캔해 Round 목록 산출"""
    names = set()
    for p in glob.glob(os.path.join(det_in_dir, "Round_*.csv")):
        bn = os.path.basename(p)
        name = bn[:-4]  # strip .csv
        if name.endswith("_raw"):
            name = name[:-4]
        # name 시작이 Round_ 인 것만
        if name.startswith("Round_"):
            names.add(name)
    return sorted(names)


def _write_passthrough_applied(round_name: str, pred_dir: str, applied_dir: str) -> str:
    """
    Similarity 스킵 시 병합이 가능하도록 applied 파일을 패스스루로 생성.
    alert_id가 없으면 __row_id를 만들어 안전하게 병합할 수 있게 함.
    """
    os.makedirs(applied_dir, exist_ok=True)
    base = os.path.join(pred_dir, f"{round_name}_with_predictions.csv")
    if not os.path.exists(base):
        raise FileNotFoundError(base)
    dfb = pd.read_csv(base, low_memory=False)

    # 병합 키: alert_id 우선, 없으면 __row_id 생성
    use_alert = "alert_id" in dfb.columns
    if not use_alert:
        dfb["__row_id"] = np.arange(len(dfb), dtype=np.int64)

    passthrough = pd.DataFrame({
        ("alert_id" if use_alert else "__row_id"): dfb["alert_id"] if use_alert else dfb["__row_id"],
        "adjusted_label": "",
        "feedback_applied": False,
        "applied_from_case": "",
        "applied_from_file": "",
        "applied_reason": "",
        "applied_confidence": "",
        "applied_similarity_score": 0.0,
        "applied_similarity_ip": 0.0,
        "applied_similarity_cosine": 0.0,
        "applied_similarity_overlap": 0.0,
        "applied_common_features": "",
    })
    out_path = os.path.join(applied_dir, f"{round_name}_position_aware_optimal.csv")
    passthrough.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="*", default=None, help="예: Round_3 Round_5")
    ap.add_argument("--all", action="store_true", help="모든 라운드 처리")

    # 디렉토리 경로
    ap.add_argument("--det-in", default="./test_rounds", help="Detection 입력 라운드 CSV 폴더")
    ap.add_argument("--pred-dir", default="./round_predictions", help="예측 결과 폴더")
    ap.add_argument("--feedback-dir", default="./feedback_cases")
    ap.add_argument("--applied-dir", default="./round_predictions_applied")
    ap.add_argument("--model-path", default="./models/xgboost_binary_classifier.joblib")

    # Detection (0단계)
    ap.add_argument("--skip-detection", action="store_true", help="Detection 단계 스킵")
    ap.add_argument("--force-detection", action="store_true", help="기존 예측이 있어도 Detection 재실행")
    ap.add_argument("--det-out", default="./round_results", help="(옵션) 결과 로그 폴더, 없어도 됨")
    ap.add_argument("--det-threshold", type=float, default=0.5)

    # Prioritizer (게이팅)
    ap.add_argument("--gate-alpha", type=float, default=0.3)
    ap.add_argument("--gate-beta", type=float, default=0.7)
    ap.add_argument("--gate-bottom-percent", type=float, default=5.0)
    ap.add_argument("--gate-top-k", type=int, default=None)
    ap.add_argument("--gate-no-shap", action="store_true")

    # Auto-feedback
    ap.add_argument("--skip-auto-feedback", action="store_true")
    ap.add_argument("--auto-top-n", type=int, default=300)
    ap.add_argument("--auto-percent", type=float, default=None)

    # Similarity
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta",  type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--no-direction", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)

    # 특정 라운드 Similarity 스킵
    ap.add_argument("--skip-similarity-rounds", nargs="*", default=[],
                    help="예: --skip-similarity-rounds Round_1 Round_3")
    ap.add_argument("--skip-similarity-round1", action="store_true",
                    help="Round_1에서만 유사사례 검색/적용을 건너뜀")

    args = ap.parse_args()

    # 라운드 결정 로직:
    # 1) --rounds가 명시되면 그걸 사용
    # 2) --all이면 pred-dir에서 찾되, 없으면 det-in에서 입력을 스캔
    if args.rounds:
        rounds = args.rounds
    elif args.all:
        rounds = list_rounds_from_predictions(args.pred_dir)
        if not rounds:
            # 아직 예측이 없다면 입력 폴더에서 라운드 목록을 만든다
            rounds = list_rounds_from_inputs(args.det_in)
    else:
        log.error("처리할 라운드가 없습니다. (--rounds ... 또는 --all)")
        return

    if not rounds:
        log.error("라운드 목록을 찾지 못했습니다. 입력/예측 폴더를 확인하세요.")
        return

    # Round_1 스킵 스위치 적용
    if args.skip_similarity_round1:
        args.skip_similarity_rounds = list(set(args.skip_similarity_rounds + ["Round_1"]))
    skip_set = set(args.skip_similarity_rounds)

    # -----------------------------
    # 0) Detection
    # -----------------------------
    if not args.skip_detection:
        log.info("=== Detection 단계 시작 ===")
        det = DetectionTool(
            rounds_directory=args.det_in,
            predictions_directory=args.pred_dir,     # 예측 csv는 pred-dir로 생성
            results_directory=args.det_out,
            model_path=args.model_path,
            rounds=rounds,                           # 선택 라운드만 생성
            threshold=args.det_threshold,
            force=args.force_detection
        ).run()
        if not det.ok:
            log.error(f"[Detection 실패] {det.message}")
            return
        log.info("=== Detection 단계 완료 ===")
    else:
        log.info("=== Detection 단계 스킵 ===")

    # -----------------------------
    # 1)~4) 라운드별 파이프라인
    # -----------------------------
    for rn in rounds:
        log.info(f"=== {rn} 시작 ===")

        # 1) Prioritizer (게이팅)
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
        if not pr.ok and pr.output_path is None:
            log.error(f"[Prioritizer 실패] {pr.message}")
            continue

        # 2) Auto-feedback (옵션) — 선택 집합 중 오탐만 자동 입력(수정된 로직 반영 파일 사용)
        if not args.skip_auto_feedback and pr.output_path:
            af = AutoFeedbackTool(
                round_name=rn,
                feedback_dir=args.feedback_dir,
                top_n=args.auto_top_n,
                percent=args.auto_percent,
            ).run()
            if not af.ok:
                log.warning(f"[AutoFeedback 경고] {af.message}")

        # 3) Similarity Apply (라운드별 스킵 지원)
        if rn in skip_set:
            log.info(f"[Similarity] {rn} 스킵 → 패스스루 applied 파일 생성")
            try:
                _ = _write_passthrough_applied(rn, args.pred_dir, args.applied_dir)
                sa_ok = True
            except Exception as e:
                log.error(f"[Similarity 스킵 패스스루 실패] {e}")
                sa_ok = False
        else:
            sa_res = SimilarityApplyTool(
                round_name=rn,
                feedback_dir=args.feedback_dir,
                out_dir=args.applied_dir,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                threshold=args.threshold,
                direction_sensitive=not args.no_direction,
                top_k=args.top_k,
            ).run()
            sa_ok = sa_res.ok
            if not sa_ok:
                log.error(f"[Similarity 실패] {sa_res.message}")

        if not sa_ok:
            continue

        # 4) Merge
        mg = MergeTool(
            round_name=rn,
            pred_dir=args.pred_dir,
            applied_dir=args.applied_dir,
        ).run()
        if not mg.ok:
            log.error(f"[Merge 실패] {mg.message}")
            continue

        log.info(f"=== {rn} 완료 ===")


if __name__ == "__main__":
    main()
