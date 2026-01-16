#!/usr/bin/env python3
"""
evaluate.py

Reads an inference CSV (output of infer.py), computes bout-level and boundary-tolerant
metrics, writes:
  - metrics.json
  - per_class.csv
and generates publication-quality plots (matplotlib, no seaborn).

Usage example:
  python scripts/evaluate.py \
    --pred_csv outputs/predictions/sc04_d1_of_tcn_predictions.csv \
    --gt_col human_labeled_state \
    --pred_col tcn_final \
    --out_dir outputs/metrics \
    --fps 30 \
    --iou_thr 0.1 \
    --tol_frames 1 3 5

Notes:
- Assumes bout_metrics.py is in scripts/ (same folder).
- Ignores frames where GT == -1 by default (configurable).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bout_metrics import bout_metrics, boundary_tolerance_score, labels_to_bouts, Bout

STATE_MAP = {
    0: "turn",
    1: "forward",
    2: "still",
    3: "explore",
    4: "rear",
    5: "groom",
    -1: "unsigned",
}
CLASSES = [0, 1, 2, 3, 4, 5]


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def framewise_confusion(gt: np.ndarray, pred: np.ndarray, classes: List[int]) -> np.ndarray:
    """Confusion matrix counts for frame-wise labels. Shapes: (K,K) where rows=GT cols=Pred."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=np.int64)
    for g, p in zip(gt, pred):
        if g in cls_to_idx and p in cls_to_idx:
            cm[cls_to_idx[g], cls_to_idx[p]] += 1
    return cm


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    """Row-normalize confusion matrix to probabilities (per-GT-class)."""
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_confusion_matrix(cm: np.ndarray, classes: List[int], out_path: Path, title: str) -> None:
    labels = [STATE_MAP[c] for c in classes]
    cmn = normalize_rows(cm)

    fig = plt.figure(figsize=(7.5, 6.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cmn, aspect="auto")  # default colormap (no manual colors)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells with % (row-normalized)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i, j]*100:.1f}%", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(per_class: Dict[int, Dict], out_path: Path, title: str) -> None:
    classes = sorted([c for c in per_class.keys() if isinstance(c, int)])
    f1 = [per_class[c]["f1"] for c in classes]
    labels = [STATE_MAP[c] for c in classes]

    fig = plt.figure(figsize=(8.0, 4.2), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(len(classes)), f1)
    ax.set_title(title)
    ax.set_ylabel("F1 (bout-level)")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_transition_tolerance_curve(scores: List[Dict], out_path: Path, title: str) -> None:
    tol = [s["tol_frames"] for s in scores]
    f1 = [s["f1"] for s in scores]
    prec = [s["precision"] for s in scores]
    rec = [s["recall"] for s in scores]

    fig = plt.figure(figsize=(7.5, 4.2), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tol, prec, marker="o", label="Precision")
    ax.plot(tol, rec, marker="o", label="Recall")
    ax.plot(tol, f1, marker="o", label="F1")
    ax.set_title(title)
    ax.set_xlabel("Tolerance (frames)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def bouts_to_duration_df(bouts: List[Bout], fps: float) -> pd.DataFrame:
    rows = []
    for b in bouts:
        rows.append(
            {
                "class_id": b.cls,
                "class": STATE_MAP.get(b.cls, str(b.cls)),
                "start_frame": b.start,
                "end_frame": b.end,
                "duration_frames": b.length,
                "duration_s": b.length / fps if fps > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_bout_duration_distributions(
    gt_bouts: List[Bout],
    pr_bouts: List[Bout],
    fps: float,
    out_path: Path,
    title: str,
    min_bouts_per_class: int = 3,
) -> None:
    """
    Publication-friendly duration comparison plot:
    For each class: show median duration (GT vs Pred) with IQR via error bars.
    (More stable than raw histograms and avoids heavy clutter.)
    """
    def summarize(bouts: List[Bout]) -> Dict[int, Tuple[float, float, float]]:
        # returns {cls: (q25, q50, q75)} in seconds
        by_cls: Dict[int, List[float]] = {c: [] for c in CLASSES}
        for b in bouts:
            if b.cls in by_cls:
                by_cls[b.cls].append(b.length / fps if fps > 0 else np.nan)
        out = {}
        for c in CLASSES:
            arr = np.array(by_cls[c], dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) >= min_bouts_per_class:
                q25, q50, q75 = np.percentile(arr, [25, 50, 75])
                out[c] = (float(q25), float(q50), float(q75))
        return out

    gt_sum = summarize(gt_bouts)
    pr_sum = summarize(pr_bouts)

    classes = [c for c in CLASSES if (c in gt_sum and c in pr_sum)]
    labels = [STATE_MAP[c] for c in classes]

    gt_med = [gt_sum[c][1] for c in classes]
    pr_med = [pr_sum[c][1] for c in classes]
    gt_err = [(gt_sum[c][1] - gt_sum[c][0], gt_sum[c][2] - gt_sum[c][1]) for c in classes]
    pr_err = [(pr_sum[c][1] - pr_sum[c][0], pr_sum[c][2] - pr_sum[c][1]) for c in classes]

    gt_err_low = [e[0] for e in gt_err]
    gt_err_high = [e[1] for e in gt_err]
    pr_err_low = [e[0] for e in pr_err]
    pr_err_high = [e[1] for e in pr_err]

    x = np.arange(len(classes))
    width = 0.38

    fig = plt.figure(figsize=(9.2, 4.6), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x - width/2, gt_med, width=width, label="GT (human)")
    ax.errorbar(x - width/2, gt_med, yerr=[gt_err_low, gt_err_high], fmt="none", capsize=3)

    ax.bar(x + width/2, pr_med, width=width, label="Pred (model)")
    ax.errorbar(x + width/2, pr_med, yerr=[pr_err_low, pr_err_high], fmt="none", capsize=3)

    ax.set_title(title)
    ax.set_ylabel("Bout duration (seconds), median ± IQR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Inference output CSV (from infer.py).")
    ap.add_argument("--gt_col", default="human_labeled_state", help="Ground truth label column in pred_csv.")
    ap.add_argument("--pred_col", default="tcn_final", help="Prediction label column in pred_csv.")
    ap.add_argument("--out_dir", default="outputs/metrics", help="Where to write metrics + plots.")
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second for duration conversion.")
    ap.add_argument("--ignore_label", type=int, default=-1, help="GT ignore label.")
    ap.add_argument("--iou_thr", type=float, default=0.1, help="IoU threshold for bout matching.")
    ap.add_argument("--tol_frames", type=int, nargs="+", default=[3], help="Transition tolerance frames list.")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    plots_dir = ensure_dir(out_dir / "plots")

    df = pd.read_csv(args.pred_csv)
    if args.gt_col not in df.columns:
        raise ValueError(f"Missing gt_col '{args.gt_col}' in {args.pred_csv}. Columns: {list(df.columns)}")
    if args.pred_col not in df.columns:
        raise ValueError(f"Missing pred_col '{args.pred_col}' in {args.pred_csv}. Columns: {list(df.columns)}")

    gt = df[args.gt_col].to_numpy(dtype=int)
    pr = df[args.pred_col].to_numpy(dtype=int)

    # Mask frames where GT is ignore_label
    keep = gt != args.ignore_label
    gt_k = gt[keep]
    pr_k = pr[keep]

    # ---------------------------
    # Metrics: bout-level
    # ---------------------------
    bm = bout_metrics(
        gt_k,
        pr_k,
        classes=CLASSES,
        ignore_labels={args.ignore_label},
        iou_thr=args.iou_thr,
    )

    # Transition tolerance metrics across multiple tolerances
    bt_scores = []
    for tol in args.tol_frames:
        bt_scores.append(
            boundary_tolerance_score(
                gt_k, pr_k,
                tol_frames=int(tol),
                ignore_labels={args.ignore_label},
            )
        )

    # Frame-wise confusion (informative, but not the main metric)
    cm = framewise_confusion(gt_k, pr_k, CLASSES)

    # ---------------------------
    # Write metrics.json
    # ---------------------------
    metrics = {
        "input": {
            "pred_csv": str(Path(args.pred_csv).resolve()),
            "gt_col": args.gt_col,
            "pred_col": args.pred_col,
            "fps": args.fps,
            "ignore_label": args.ignore_label,
            "iou_thr": args.iou_thr,
            "tol_frames": args.tol_frames,
            "n_frames_total": int(len(gt)),
            "n_frames_used": int(len(gt_k)),
        },
        "bout_metrics": bm,
        "transition_tolerance": bt_scores,
        "frame_confusion_counts": cm.tolist(),
        "frame_confusion_row_normalized": normalize_rows(cm).tolist(),
    }
    save_json(metrics, out_dir / "metrics.json")

    # ---------------------------
    # Write per_class.csv
    # ---------------------------
    per_class_rows = []
    for c in CLASSES:
        d = bm["per_class"].get(c, {})
        per_class_rows.append(
            {
                "class_id": c,
                "class": STATE_MAP[c],
                "gt_bouts": d.get("gt_bouts", 0),
                "pred_bouts": d.get("pred_bouts", 0),
                "TP": d.get("TP", 0),
                "FP": d.get("FP", 0),
                "FN": d.get("FN", 0),
                "precision": d.get("precision", 0.0),
                "recall": d.get("recall", 0.0),
                "f1": d.get("f1", 0.0),
                "mean_iou": d.get("mean_iou", 0.0),
            }
        )
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(out_dir / "per_class.csv", index=False)

    # ---------------------------
    # Plots (publication-friendly)
    # ---------------------------
    # 1) Per-class bout F1
    plot_per_class_f1(
        bm["per_class"],
        plots_dir / "per_class_bout_f1.png",
        title="Bout-level F1 by Class",
    )

    # 2) Frame-wise confusion matrix (row-normalized)
    plot_confusion_matrix(
        cm,
        CLASSES,
        plots_dir / "frame_confusion_matrix.png",
        title="Frame-wise Confusion Matrix (Row-normalized)",
    )

    # 3) Transition tolerance curve
    plot_transition_tolerance_curve(
        bt_scores,
        plots_dir / "transition_tolerance_curve.png",
        title="Transition Detection vs Tolerance",
    )

    # 4) Bout duration comparison (median ± IQR)
    gt_bouts = labels_to_bouts(gt_k, ignore_labels={args.ignore_label})
    pr_bouts = labels_to_bouts(pr_k, ignore_labels={args.ignore_label})
    plot_bout_duration_distributions(
        gt_bouts,
        pr_bouts,
        fps=args.fps,
        out_path=plots_dir / "bout_duration_median_iqr.png",
        title="Bout Duration Comparison (Median ± IQR)",
    )

    # Optional: export bout tables (helpful for debugging/review)
    bouts_to_duration_df(gt_bouts, args.fps).to_csv(out_dir / "gt_bouts.csv", index=False)
    bouts_to_duration_df(pr_bouts, args.fps).to_csv(out_dir / "pred_bouts.csv", index=False)

    # Summary to console
    overall = bm["overall"]
    print("[OK] Wrote:")
    print(f"  - {out_dir / 'metrics.json'}")
    print(f"  - {out_dir / 'per_class.csv'}")
    print(f"  - plots in {plots_dir}")
    print("\n[Summary] Bout-level overall:")
    print(f"  precision={overall['precision']:.3f} recall={overall['recall']:.3f} f1={overall['f1']:.3f}")
    print(f"  mean_iou={overall['mean_iou']:.3f} weighted_mean_iou={overall['weighted_mean_iou']:.3f}")
    for s in bt_scores:
        print(f"[Transitions] tol={s['tol_frames']} frames: precision={s['precision']:.3f} "
              f"recall={s['recall']:.3f} f1={s['f1']:.3f}")


if __name__ == "__main__":
    main()
