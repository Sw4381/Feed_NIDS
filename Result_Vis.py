# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def analyze_feedback_effect(applied_dir="./round_predictions_applied"):
    files = sorted(glob.glob(os.path.join(applied_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"피드백 적용 파일 없음: {applied_dir}")

    results = []
    for f in files:
        round_name = os.path.basename(f).replace(".csv", "")
        print(f"\n[{round_name}] 분석 시작")
        df = pd.read_csv(f, low_memory=False)

        if "label" not in df.columns:
            print("⚠️ 실제 라벨(label) 없음 → 스킵")
            continue

        # 적용 전/후 라벨
        y_true = df["label"].astype(str)
        y_pred_before = df["predicted_label"].astype(str)
        y_pred_after  = df.get("adjusted_label", y_pred_before).astype(str)

        # 메트릭 계산 함수
        def metrics(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=["Normal","Attack"], average="binary", pos_label="Attack"
            )
            return acc, prec, rec, f1

        acc_b, prec_b, rec_b, f1_b = metrics(y_true, y_pred_before)
        acc_a, prec_a, rec_a, f1_a = metrics(y_true, y_pred_after)

        # confusion matrix
        cm_before = confusion_matrix(y_true, y_pred_before, labels=["Normal","Attack"])
        cm_after  = confusion_matrix(y_true, y_pred_after, labels=["Normal","Attack"])

        # 피드백 적용된 건수
        n_applied = int(df["feedback_applied"].sum())
        n_corrected = int(((y_pred_before != y_true) & (y_pred_after == y_true)).sum())
        n_misapplied = int(((y_pred_before == y_true) & (y_pred_after != y_true)).sum())

        # 저장
        results.append({
            "round": round_name,
            "acc_before": acc_b, "acc_after": acc_a,
            "prec_before": prec_b, "prec_after": prec_a,
            "rec_before": rec_b, "rec_after": rec_a,
            "f1_before": f1_b, "f1_after": f1_a,
            "applied": n_applied,
            "corrected": n_corrected,
            "misapplied": n_misapplied
        })

        # 출력 요약
        print(f"  적용 전 정확도: {acc_b:.4f}, 적용 후 정확도: {acc_a:.4f} (Δ {acc_a-acc_b:+.4f})")
        print(f"  Precision: {prec_b:.3f} → {prec_a:.3f}")
        print(f"  Recall:    {rec_b:.3f} → {rec_a:.3f}")
        print(f"  F1:        {f1_b:.3f} → {f1_a:.3f}")
        print(f"  ConfusionMatrix Before:\n{cm_before}")
        print(f"  ConfusionMatrix After:\n{cm_after}")
        print(f"  적용된 샘플: {n_applied}, 올바르게 수정: {n_corrected}, 잘못 수정: {n_misapplied}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    summary_df = analyze_feedback_effect("./round_predictions_applied")
    print("\n=== 전체 라운드 요약 ===")
    print(summary_df)
