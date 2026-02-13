#!/usr/bin/env python3
"""
evaluate.py

Reads inference CSV(s) (output of infer.py), computes bout-level and boundary-tolerant
metrics, writes metrics + publication-quality plots.

Supports two input modes:
  1. Single CSV (original):
       --pred_csv outputs/predictions/sc04_d1_of_predictions.csv

  2. Batch from directory (matches infer.py --out_dir):
       --pred_dir outputs/predictions/
     Evaluates every *_predictions.csv in the directory.
     Use --session_prefix to filter to specific sessions.

Outputs (per session + aggregate):
  - metrics.json           (per-session or aggregate)
  - per_class.csv
  - plots/                 (confusion matrix, F1 bars, tolerance curve, duration comparison)

Usage examples:
  # Single session (backward-compatible)
  python evaluate.py --pred_csv outputs/predictions/sc04_d1_of_predictions.csv \
    --out_dir outputs/metrics

  # Batch: evaluate all sessions
  python evaluate.py --pred_dir outputs/predictions/ \
    --out_dir outputs/metrics

  # Batch: only sessions matching prefix
  python evaluate.py --pred_dir outputs/predictions/ \
    --out_dir outputs/metrics --session_prefix sc04

Notes:
- Assumes bout_metrics.py is importable (same folder or on PYTHONPATH).
- Ignores frames where GT == -1 by default (configurable).
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ============================================================
# Utilities
# ============================================================

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


# ============================================================
# Plotting
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, classes: List[int], out_path: Path, title: str) -> None:
    labels = [STATE_MAP[c] for c in classes]
    cmn = normalize_rows(cm)

    fig = plt.figure(figsize=(7.5, 6.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cmn, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

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


def bouts_to_duration_df(bouts: List[Bout], fps: float,
                         session_prefix: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for b in bouts:
        row = {
            "class_id": b.cls,
            "class": STATE_MAP.get(b.cls, str(b.cls)),
            "start_frame": b.start,
            "end_frame": b.end,
            "duration_frames": b.length,
            "duration_s": b.length / fps if fps > 0 else np.nan,
        }
        if session_prefix is not None:
            row["session_prefix"] = session_prefix
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bout_duration_distributions(
    gt_bouts: List[Bout],
    pr_bouts: List[Bout],
    fps: float,
    out_path: Path,
    title: str,
    min_bouts_per_class: int = 3,
) -> None:
    def summarize(bouts: List[Bout]) -> Dict[int, Tuple[float, float, float]]:
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


def plot_cross_session_f1_comparison(
    all_session_metrics: Dict[str, Dict],
    out_path: Path,
    title: str = "Per-class Bout F1 across Sessions",
) -> None:
    """Grouped bar chart: one group per behavior class, one bar per session."""
    sessions = sorted(all_session_metrics.keys())
    if not sessions:
        return

    n_sessions = len(sessions)
    n_classes = len(CLASSES)
    labels = [STATE_MAP[c] for c in CLASSES]

    fig = plt.figure(figsize=(max(9, n_sessions * 1.5), 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    width = 0.8 / n_sessions
    x = np.arange(n_classes)

    for i, sess in enumerate(sessions):
        pc = all_session_metrics[sess].get("per_class", {})
        f1_vals = []
        for c in CLASSES:
            d = pc.get(c, pc.get(str(c), {}))
            f1_vals.append(d.get("f1", 0.0))
        offset = (i - n_sessions / 2 + 0.5) * width
        ax.bar(x + offset, f1_vals, width=width, label=sess)

    ax.set_title(title)
    ax.set_ylabel("F1 (bout-level)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=min(n_sessions, 4))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_session_overall_summary(
    all_session_metrics: Dict[str, Dict],
    out_path: Path,
    title: str = "Overall Bout-level Metrics by Session",
) -> None:
    """Bar chart of overall precision, recall, F1 per session."""
    sessions = sorted(all_session_metrics.keys())
    if not sessions:
        return

    prec = []
    rec = []
    f1 = []
    for sess in sessions:
        ov = all_session_metrics[sess].get("overall", {})
        prec.append(ov.get("precision", 0.0))
        rec.append(ov.get("recall", 0.0))
        f1.append(ov.get("f1", 0.0))

    x = np.arange(len(sessions))
    width = 0.25

    fig = plt.figure(figsize=(max(8, len(sessions) * 1.2), 4.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x - width, prec, width=width, label="Precision")
    ax.bar(x, rec, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Core evaluation for one session's arrays
# ============================================================

def evaluate_arrays(
    gt: np.ndarray,
    pr: np.ndarray,
    ignore_label: int,
    iou_thr: float,
    tol_frames: List[int],
    fps: float,
    session_prefix: Optional[str] = None,
) -> Dict:
    """Compute all metrics for a single gt/pred pair. Returns a dict of results."""
    keep = gt != ignore_label
    gt_k = gt[keep]
    pr_k = pr[keep]

    bm = bout_metrics(
        gt_k, pr_k,
        classes=CLASSES,
        ignore_labels={ignore_label},
        iou_thr=iou_thr,
    )

    bt_scores = []
    for tol in tol_frames:
        bt_scores.append(
            boundary_tolerance_score(
                gt_k, pr_k,
                tol_frames=int(tol),
                ignore_labels={ignore_label},
            )
        )

    cm = framewise_confusion(gt_k, pr_k, CLASSES)

    gt_bouts = labels_to_bouts(gt_k, ignore_labels={ignore_label})
    pr_bouts = labels_to_bouts(pr_k, ignore_labels={ignore_label})

    return {
        "gt_k": gt_k,
        "pr_k": pr_k,
        "bout_metrics": bm,
        "transition_tolerance": bt_scores,
        "confusion_matrix": cm,
        "gt_bouts": gt_bouts,
        "pr_bouts": pr_bouts,
        "n_frames_total": int(len(gt)),
        "n_frames_used": int(len(gt_k)),
        "session_prefix": session_prefix,
    }


# ============================================================
# Write outputs for one evaluation result
# ============================================================

def write_session_outputs(
    result: Dict,
    out_dir: Path,
    args,
    pred_csv_path: str,
    title_prefix: str = "",
) -> None:
    """Write metrics.json, per_class.csv, and plots for one evaluation result."""
    plots_dir = ensure_dir(out_dir / "plots")

    bm = result["bout_metrics"]
    bt_scores = result["transition_tolerance"]
    cm = result["confusion_matrix"]
    gt_bouts = result["gt_bouts"]
    pr_bouts = result["pr_bouts"]

    # metrics.json
    metrics = {
        "input": {
            "pred_csv": str(Path(pred_csv_path).resolve()),
            "session_prefix": result.get("session_prefix"),
            "gt_col": args.gt_col,
            "pred_col": args.pred_col,
            "fps": args.fps,
            "ignore_label": args.ignore_label,
            "iou_thr": args.iou_thr,
            "tol_frames": args.tol_frames,
            "n_frames_total": result["n_frames_total"],
            "n_frames_used": result["n_frames_used"],
        },
        "bout_metrics": bm,
        "transition_tolerance": bt_scores,
        "frame_confusion_counts": cm.tolist(),
        "frame_confusion_row_normalized": normalize_rows(cm).tolist(),
    }
    save_json(metrics, out_dir / "metrics.json")

    # per_class.csv
    per_class_rows = []
    for c in CLASSES:
        d = bm["per_class"].get(c, {})
        per_class_rows.append({
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
        })
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class.csv", index=False)

    # Plots
    tp = f"{title_prefix} — " if title_prefix else ""

    plot_per_class_f1(
        bm["per_class"],
        plots_dir / "per_class_bout_f1.png",
        title=f"{tp}Bout-level F1 by Class",
    )
    plot_confusion_matrix(
        cm, CLASSES,
        plots_dir / "frame_confusion_matrix.png",
        title=f"{tp}Frame-wise Confusion Matrix (Row-normalized)",
    )
    plot_transition_tolerance_curve(
        bt_scores,
        plots_dir / "transition_tolerance_curve.png",
        title=f"{tp}Transition Detection vs Tolerance",
    )
    plot_bout_duration_distributions(
        gt_bouts, pr_bouts,
        fps=args.fps,
        out_path=plots_dir / "bout_duration_median_iqr.png",
        title=f"{tp}Bout Duration Comparison (Median ± IQR)",
    )

    # Bout tables
    bouts_to_duration_df(gt_bouts, args.fps, result.get("session_prefix")).to_csv(
        out_dir / "gt_bouts.csv", index=False)
    bouts_to_duration_df(pr_bouts, args.fps, result.get("session_prefix")).to_csv(
        out_dir / "pred_bouts.csv", index=False)

    # Console summary
    overall = bm["overall"]
    print(f"  precision={overall['precision']:.3f}  recall={overall['recall']:.3f}  "
          f"f1={overall['f1']:.3f}  mean_iou={overall['mean_iou']:.3f}")
    for s in bt_scores:
        print(f"  [tol={s['tol_frames']}] prec={s['precision']:.3f} "
              f"rec={s['recall']:.3f} f1={s['f1']:.3f}")


# ============================================================
# Discover prediction CSVs
# ============================================================

def discover_pred_csvs(
    pred_dir: str,
    session_prefix_filter: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Find *_predictions.csv files in pred_dir.
    Returns list of (session_prefix, csv_path) tuples.

    Session prefix is derived by stripping '_predictions.csv' from the filename.
    If session_prefix_filter is given, only matching prefixes are returned.
    """
    pred_dir = Path(pred_dir)
    csvs = sorted(pred_dir.glob("*_predictions.csv"))

    if not csvs:
        # Fallback: try all CSVs
        csvs = sorted(pred_dir.glob("*.csv"))

    results = []
    for csv_path in csvs:
        stem = csv_path.stem
        # Strip _predictions suffix to get session prefix
        if stem.endswith("_predictions"):
            prefix = stem[: -len("_predictions")]
        else:
            prefix = stem

        if session_prefix_filter is not None:
            filt = session_prefix_filter.strip()
            if not (prefix == filt or prefix.startswith(filt)):
                continue

        results.append((prefix, str(csv_path)))

    if not results:
        raise FileNotFoundError(
            f"No prediction CSVs found in {pred_dir}"
            + (f" matching prefix '{session_prefix_filter}'" if session_prefix_filter else "")
        )

    return results


# ============================================================
# Aggregate metrics across sessions
# ============================================================

def aggregate_results(
    all_results: Dict[str, Dict],
    args,
    out_dir: Path,
) -> None:
    """
    Pool GT/pred arrays from all sessions and compute aggregate metrics.
    Also produce cross-session comparison plots.
    """
    agg_dir = ensure_dir(out_dir / "aggregate")
    plots_dir = ensure_dir(agg_dir / "plots")

    # Pool arrays
    gt_all = np.concatenate([r["gt_k"] for r in all_results.values()])
    pr_all = np.concatenate([r["pr_k"] for r in all_results.values()])

    # Aggregate bout-level metrics on pooled data
    bm = bout_metrics(
        gt_all, pr_all,
        classes=CLASSES,
        ignore_labels={args.ignore_label},
        iou_thr=args.iou_thr,
    )

    bt_scores = []
    for tol in args.tol_frames:
        bt_scores.append(
            boundary_tolerance_score(
                gt_all, pr_all,
                tol_frames=int(tol),
                ignore_labels={args.ignore_label},
            )
        )

    cm = framewise_confusion(gt_all, pr_all, CLASSES)
    gt_bouts = labels_to_bouts(gt_all, ignore_labels={args.ignore_label})
    pr_bouts = labels_to_bouts(pr_all, ignore_labels={args.ignore_label})

    # Write aggregate metrics.json
    sessions_list = sorted(all_results.keys())
    metrics = {
        "input": {
            "mode": "aggregate",
            "sessions": sessions_list,
            "n_sessions": len(sessions_list),
            "gt_col": args.gt_col,
            "pred_col": args.pred_col,
            "fps": args.fps,
            "ignore_label": args.ignore_label,
            "iou_thr": args.iou_thr,
            "tol_frames": args.tol_frames,
            "n_frames_total": sum(r["n_frames_total"] for r in all_results.values()),
            "n_frames_used": int(len(gt_all)),
        },
        "bout_metrics": bm,
        "transition_tolerance": bt_scores,
        "frame_confusion_counts": cm.tolist(),
        "frame_confusion_row_normalized": normalize_rows(cm).tolist(),
        "per_session_overall": {
            sess: r["bout_metrics"]["overall"] for sess, r in all_results.items()
        },
    }
    save_json(metrics, agg_dir / "metrics.json")

    # Aggregate per_class.csv
    per_class_rows = []
    for c in CLASSES:
        d = bm["per_class"].get(c, {})
        per_class_rows.append({
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
        })
    pd.DataFrame(per_class_rows).to_csv(agg_dir / "per_class.csv", index=False)

    # Aggregate plots
    plot_per_class_f1(
        bm["per_class"],
        plots_dir / "per_class_bout_f1.png",
        title="Aggregate — Bout-level F1 by Class",
    )
    plot_confusion_matrix(
        cm, CLASSES,
        plots_dir / "frame_confusion_matrix.png",
        title="Aggregate — Frame-wise Confusion Matrix",
    )
    plot_transition_tolerance_curve(
        bt_scores,
        plots_dir / "transition_tolerance_curve.png",
        title="Aggregate — Transition Detection vs Tolerance",
    )
    plot_bout_duration_distributions(
        gt_bouts, pr_bouts,
        fps=args.fps,
        out_path=plots_dir / "bout_duration_median_iqr.png",
        title="Aggregate — Bout Duration Comparison (Median ± IQR)",
    )

    # Cross-session comparison plots
    session_bm = {sess: r["bout_metrics"] for sess, r in all_results.items()}

    plot_cross_session_f1_comparison(
        session_bm,
        plots_dir / "cross_session_f1_comparison.png",
        title="Per-class Bout F1 across Sessions",
    )
    plot_session_overall_summary(
        session_bm,
        plots_dir / "session_overall_summary.png",
        title="Overall Bout-level Metrics by Session",
    )

    # Bout tables (pooled)
    bouts_to_duration_df(gt_bouts, args.fps).to_csv(agg_dir / "gt_bouts.csv", index=False)
    bouts_to_duration_df(pr_bouts, args.fps).to_csv(agg_dir / "pred_bouts.csv", index=False)

    # Summary table: one row per session
    summary_rows = []
    for sess in sessions_list:
        ov = all_results[sess]["bout_metrics"]["overall"]
        summary_rows.append({
            "session": sess,
            "n_frames": all_results[sess]["n_frames_used"],
            "precision": ov.get("precision", 0.0),
            "recall": ov.get("recall", 0.0),
            "f1": ov.get("f1", 0.0),
            "mean_iou": ov.get("mean_iou", 0.0),
        })
    pd.DataFrame(summary_rows).to_csv(agg_dir / "session_summary.csv", index=False)

    # Console
    overall = bm["overall"]
    print(f"\n{'='*55}")
    print(f"[Aggregate] {len(sessions_list)} sessions, {len(gt_all)} frames")
    print(f"  precision={overall['precision']:.3f}  recall={overall['recall']:.3f}  "
          f"f1={overall['f1']:.3f}  mean_iou={overall['mean_iou']:.3f}")
    for s in bt_scores:
        print(f"  [tol={s['tol_frames']}] prec={s['precision']:.3f} "
              f"rec={s['recall']:.3f} f1={s['f1']:.3f}")
    print(f"[Aggregate] Wrote: {agg_dir}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate TCN predictions — single CSV or batch from directory."
    )

    # --- Input mode: single ---
    ap.add_argument("--pred_csv", default=None,
                    help="Single inference CSV (from infer.py).")

    # --- Input mode: batch ---
    ap.add_argument("--pred_dir", default=None,
                    help="Directory of *_predictions.csv files (from infer.py batch mode). "
                         "Evaluates each session and produces aggregate metrics.")

    # --- Shared ---
    ap.add_argument("--session_prefix", default=None,
                    help="Optional filter. In single mode: used as plot title prefix. "
                         "In batch mode: only evaluate sessions whose prefix matches.")
    ap.add_argument("--gt_col", default="human_labeled_state",
                    help="Ground truth label column.")
    ap.add_argument("--pred_col", default="tcn_final",
                    help="Prediction label column.")
    ap.add_argument("--out_dir", default="outputs/metrics",
                    help="Where to write metrics + plots.")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Frames per second for duration conversion.")
    ap.add_argument("--ignore_label", type=int, default=-1,
                    help="GT ignore label.")
    ap.add_argument("--iou_thr", type=float, default=0.1,
                    help="IoU threshold for bout matching.")
    ap.add_argument("--tol_frames", type=int, nargs="+", default=[3],
                    help="Transition tolerance frames list.")

    args = ap.parse_args()

    if args.pred_csv is None and args.pred_dir is None:
        raise ValueError("Provide either --pred_csv (single) or --pred_dir (batch).")

    out_dir = ensure_dir(args.out_dir)

    # ===========================================
    # Single-session mode
    # ===========================================
    if args.pred_csv is not None and args.pred_dir is None:
        print(f"[Mode] Single CSV: {args.pred_csv}")

        df = pd.read_csv(args.pred_csv)
        if args.gt_col not in df.columns:
            raise ValueError(f"Missing gt_col '{args.gt_col}'. Columns: {list(df.columns)}")
        if args.pred_col not in df.columns:
            raise ValueError(f"Missing pred_col '{args.pred_col}'. Columns: {list(df.columns)}")

        gt = df[args.gt_col].to_numpy(dtype=int)
        pr = df[args.pred_col].to_numpy(dtype=int)

        prefix = args.session_prefix
        # Try to detect from the session_prefix column if present
        if prefix is None and "session_prefix" in df.columns:
            prefix = str(df["session_prefix"].iloc[0])

        result = evaluate_arrays(
            gt, pr,
            ignore_label=args.ignore_label,
            iou_thr=args.iou_thr,
            tol_frames=args.tol_frames,
            fps=args.fps,
            session_prefix=prefix,
        )

        title_prefix = prefix or Path(args.pred_csv).stem
        print(f"\n[Eval] {title_prefix}")
        write_session_outputs(result, out_dir, args, args.pred_csv, title_prefix)
        print(f"\n[OK] Outputs in {out_dir}")

    # ===========================================
    # Batch mode
    # ===========================================
    else:
        pred_dir = args.pred_dir or (str(Path(args.pred_csv).parent) if args.pred_csv else None)
        if pred_dir is None:
            raise ValueError("Cannot determine prediction directory.")

        print(f"[Mode] Batch from: {pred_dir}")
        session_csvs = discover_pred_csvs(pred_dir, args.session_prefix)

        filter_msg = (f" (filtered by '{args.session_prefix}')" if args.session_prefix else "")
        print(f"[Found] {len(session_csvs)} session(s){filter_msg}:")
        for prefix, csv_path in session_csvs:
            print(f"  - {prefix}: {csv_path}")

        all_results = {}
        errors = []

        for prefix, csv_path in session_csvs:
            print(f"\n[Eval] {prefix}")
            try:
                df = pd.read_csv(csv_path)
                if args.gt_col not in df.columns:
                    raise ValueError(f"Missing gt_col '{args.gt_col}'. Columns: {list(df.columns)}")
                if args.pred_col not in df.columns:
                    raise ValueError(f"Missing pred_col '{args.pred_col}'. Columns: {list(df.columns)}")

                gt = df[args.gt_col].to_numpy(dtype=int)
                pr = df[args.pred_col].to_numpy(dtype=int)

                result = evaluate_arrays(
                    gt, pr,
                    ignore_label=args.ignore_label,
                    iou_thr=args.iou_thr,
                    tol_frames=args.tol_frames,
                    fps=args.fps,
                    session_prefix=prefix,
                )

                # Per-session output directory
                sess_dir = ensure_dir(out_dir / prefix)
                write_session_outputs(result, sess_dir, args, csv_path, prefix)
                all_results[prefix] = result

            except Exception as e:
                print(f"  [ERROR] {prefix}: {e}")
                errors.append((prefix, str(e)))

        # Aggregate across sessions
        if len(all_results) > 1:
            aggregate_results(all_results, args, out_dir)
        elif len(all_results) == 1:
            print("\n[Note] Only 1 session — skipping cross-session aggregate.")

        # Final summary
        print(f"\n{'='*55}")
        print(f"[Done] {len(all_results)}/{len(session_csvs)} sessions evaluated.")
        if errors:
            print(f"[Errors] {len(errors)} failed:")
            for prefix, err in errors:
                print(f"  - {prefix}: {err}")
        print(f"[Outputs] {out_dir}")


if __name__ == "__main__":
    main()