#!/usr/bin/env python3
"""
evaluate.py — Publication-standard evaluation for temporal behavior segmentation

Metrics follow conventions from the temporal action segmentation (TAS) and
behavioral neuroscience (DLC, SimBA, MARS, B-SOiD) communities:

  Frame-level:
    - Accuracy (Mean over Frames / MoF)
    - Per-class precision, recall, F1, macro-F1, weighted-F1
    - Cohen's kappa (model treated as inter-rater vs human annotation)
    - Collar accuracy: frame accuracy after excluding ±K frames around
      every GT transition boundary — standard approach to handle annotation
      ambiguity at state boundaries (configurable with --collar)

  Segment-level (standard TAS metrics):
    - F1@{10, 25, 50}: segment-level F1 at IoU thresholds 0.10, 0.25, 0.50
      A predicted segment is TP if IoU with a same-class GT segment > τ.
      (Lea et al. 2017; Farha & Gall 2019; Yi et al. 2021)
    - Edit Score: normalized Levenshtein distance on the ordered segment
      sequence, measuring whether segment ordering is correct regardless of
      exact boundary placement (Lea et al. 2017)

  Bout-level (from bout_metrics.py):
    - Bout-matched precision, recall, F1, mean IoU
    - Boundary tolerance P/R/F1 at configurable tolerances
    - Bout duration distributions (median ± IQR)

Usage:
  # Single session
  python evaluate.py --pred_csv outputs/predictions/sc04_predictions.csv \\
    --out_dir outputs/metrics

  # Batch
  python evaluate.py --pred_dir outputs/predictions/ --out_dir outputs/metrics

  # Custom collar (default ±5 frames = ±0.17s at 30fps)
  python evaluate.py --pred_dir outputs/predictions/ --out_dir outputs/metrics \\
    --collar 5 10 15
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bout_metrics import bout_metrics, boundary_tolerance_score, labels_to_bouts, Bout

STATE_MAP = {
    0: "turn", 1: "forward", 2: "still",
    3: "explore", 4: "rear", 5: "groom",
    -1: "unsigned",
}
CLASSES = [0, 1, 2, 3, 4, 5]
N_CLASSES = len(CLASSES)


# ============================================================
# Majority vote smoothing
# ============================================================

def majority_vote(labels: np.ndarray, win: int, ignore_label: int = -1) -> np.ndarray:
    """
    Sliding-window majority vote on label sequence.
    Window is centered: ±(win//2) frames around each frame.
    Ties broken by keeping the original label.
    """
    if win < 3:
        return labels.copy()
    # Ensure odd window for symmetric centering
    if win % 2 == 0:
        win += 1
    r = win // 2
    T = len(labels)
    out = labels.copy()
    for t in range(T):
        a = max(0, t - r)
        b = min(T, t + r + 1)
        seg = labels[a:b]
        seg = seg[seg != ignore_label]
        if len(seg) == 0:
            continue
        vals, counts = np.unique(seg, return_counts=True)
        best = vals[np.argmax(counts)]
        # Break ties: if original label ties for max, keep it
        orig = labels[t]
        if orig != ignore_label and orig in vals:
            orig_count = counts[vals == orig][0]
            if orig_count == counts.max():
                best = orig
        out[t] = best
    return out


# ============================================================
# Utilities
# ============================================================

def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# GT label correction: enforce stillness by speed
# ============================================================

def enforce_stillness_by_speed(y: np.ndarray, speed: np.ndarray,
                               threshold: float = 0.1,
                               min_duration: int = 30,
                               still_label: int = 2) -> np.ndarray:
    """
    Override GT labels to 'still' (label 2) when speed remains below a
    threshold for a continuous stretch longer than min_duration frames.

    This corrects annotation errors where a stationary mouse was labeled
    as e.g. 'explore' or 'turn' despite near-zero locomotion.

    Args:
        y:             Label array (modified in-place on a copy).
        speed:         Speed array (cm/s), same length as y.
        threshold:     Speed cutoff in cm/s (default 0.1).
        min_duration:  Minimum consecutive low-speed frames to trigger
                       override (default 30 = 1s at 30fps).
        still_label:   Integer label for 'still' (default 2).

    Returns:
        Corrected label array (copy).
    """
    n = min(len(y), len(speed))
    y_out = y[:n].copy()
    speed = speed[:n]

    # Boolean mask: True where speed is below threshold
    low = speed < threshold

    # Find contiguous runs of low speed using diff on padded array
    padded = np.concatenate(([False], low, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    frames_modified = 0
    for s, e in zip(starts, ends):
        if (e - s) > min_duration:
            changed = np.sum(y_out[s:e] != still_label)
            frames_modified += changed
            y_out[s:e] = still_label

    if frames_modified > 0:
        print(f"  [STILL] Enforced stillness (speed < {threshold} cm/s "
              f"for > {min_duration} frames) on {frames_modified} frames")

    return y_out


# ============================================================
# Frame-level metrics
# ============================================================

def framewise_confusion(gt, pred, classes=CLASSES):
    """Confusion matrix: rows=GT, cols=Pred."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=np.int64)
    for g, p in zip(gt, pred):
        if g in cls_to_idx and p in cls_to_idx:
            cm[cls_to_idx[g], cls_to_idx[p]] += 1
    return cm


def normalize_rows(cm):
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def framewise_metrics(gt, pred, classes=CLASSES):
    """Per-class P/R/F1, accuracy, macro-F1, weighted-F1."""
    cm = framewise_confusion(gt, pred, classes)
    results = {}
    f1s, weights = [], []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        n = cm[i, :].sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[c] = {"precision": p, "recall": r, "f1": f1, "support": int(n)}
        f1s.append(f1)
        weights.append(n)

    accuracy = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0
    macro_f1 = np.mean(f1s)
    w = np.array(weights, dtype=np.float64)
    weighted_f1 = np.average(f1s, weights=w) if w.sum() > 0 else 0.0

    return {
        "per_class": results,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
    }


def cohens_kappa(cm):
    """
    Cohen's kappa from confusion matrix.
    Treats model predictions as a second "rater" vs human annotations.
    Interpretation (Landis & Koch 1977):
      <0.20 poor, 0.21–0.40 fair, 0.41–0.60 moderate,
      0.61–0.80 substantial, 0.81–1.00 almost perfect
    """
    cm = cm.astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0
    p_o = cm.trace() / n  # observed agreement
    # Expected agreement under independence
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (n * n)
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def kappa_interpretation(k):
    if k < 0.0:
        return "poor"
    elif k <= 0.20:
        return "slight"
    elif k <= 0.40:
        return "fair"
    elif k <= 0.60:
        return "moderate"
    elif k <= 0.80:
        return "substantial"
    else:
        return "almost perfect"


# ============================================================
# Transition collar masking
# ============================================================

def find_transition_frames(labels):
    """Find indices where label changes (transition boundaries)."""
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    return transitions


def build_collar_mask(labels, collar_frames):
    """
    Build boolean mask that is True for frames OUTSIDE the collar zone.
    Frames within ±collar_frames of any GT transition are masked out.
    This handles annotation ambiguity at behavior boundaries.
    """
    T = len(labels)
    mask = np.ones(T, dtype=bool)
    transitions = find_transition_frames(labels)
    for t in transitions:
        lo = max(0, t - collar_frames)
        hi = min(T, t + collar_frames + 1)
        mask[lo:hi] = False
    return mask


def collar_accuracy(gt, pred, collar_frames):
    """Frame accuracy after excluding ±collar_frames around GT transitions."""
    mask = build_collar_mask(gt, collar_frames)
    gt_c = gt[mask]
    pr_c = pred[mask]
    if len(gt_c) == 0:
        return 0.0, 0
    acc = (gt_c == pr_c).mean()
    return float(acc), int(mask.sum())


# ============================================================
# Segment-level metrics (TAS standard)
# ============================================================

def labels_to_segments(labels):
    """
    Convert frame-level labels to ordered list of (class, start, end) segments.
    Each segment is a contiguous run of the same label.
    """
    if len(labels) == 0:
        return []
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((int(labels[start]), start, i))
            start = i
    segments.append((int(labels[start]), start, len(labels)))
    return segments


def segment_iou(seg_a, seg_b):
    """IoU between two segments (class, start, end)."""
    _, s1, e1 = seg_a
    _, s2, e2 = seg_b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def segment_f1_at_iou(gt_labels, pred_labels, iou_threshold):
    """
    Segment-level F1 at a given IoU threshold (F1@τ).
    Standard TAS metric (Lea et al. 2017, Farha & Gall 2019).

    For each class independently: predicted segments are matched to GT
    segments. A prediction is TP if IoU > threshold with a same-class GT
    segment. Each GT segment can be matched at most once.
    """
    gt_segs = labels_to_segments(gt_labels)
    pred_segs = labels_to_segments(pred_labels)

    tp_total, fp_total, fn_total = 0, 0, 0

    for c in CLASSES:
        gt_c = [s for s in gt_segs if s[0] == c]
        pred_c = [s for s in pred_segs if s[0] == c]

        matched_gt = set()
        tp, fp = 0, 0

        for ps in pred_c:
            best_iou = 0.0
            best_idx = -1
            for j, gs in enumerate(gt_c):
                if j in matched_gt:
                    continue
                iou = segment_iou(ps, gs)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gt_c) - len(matched_gt)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
    }


def edit_score(gt_labels, pred_labels):
    """
    Normalized Edit Score (Lea et al. 2017).
    Levenshtein distance on the ordered sequence of segment class labels,
    normalized to [0, 100]. Measures ordering correctness independent of
    exact boundary placement.

    Score = (1 - edit_distance / max(|gt_segs|, |pred_segs|)) * 100
    Higher is better. 100 = perfect segment ordering.
    """
    gt_segs = labels_to_segments(gt_labels)
    pred_segs = labels_to_segments(pred_labels)

    gt_seq = [s[0] for s in gt_segs]
    pred_seq = [s[0] for s in pred_segs]

    # Levenshtein distance (dynamic programming)
    n, m = len(gt_seq), len(pred_seq)
    if n == 0 and m == 0:
        return 100.0
    if n == 0 or m == 0:
        return 0.0

    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_seq[i - 1] == pred_seq[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    dist = dp[n, m]
    score = (1.0 - dist / max(n, m)) * 100.0
    return float(max(0.0, score))


# ============================================================
# Plotting
# ============================================================

def plot_confusion_matrix(cm, classes, out_path, title):
    labels = [STATE_MAP[c] for c in classes]
    cmn = normalize_rows(cm)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200)
    im = ax.imshow(cmn, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            color = "white" if cmn[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cmn[i, j]*100:.1f}%", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rate")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(per_class, out_path, title, metric_key="f1"):
    classes = sorted([c for c in per_class.keys() if isinstance(c, int)])
    f1 = [per_class[c].get(metric_key, 0.0) for c in classes]
    labels = [STATE_MAP[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    bars = ax.bar(np.arange(len(classes)), f1, color="#4C72B0")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("F1")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_segment_f1_bars(f1_at_ious, out_path, title):
    """Bar chart of F1@10, F1@25, F1@50 + Edit Score."""
    labels = [f"F1@{int(r['iou_threshold']*100)}" for r in f1_at_ious]
    values = [r["f1"] for r in f1_at_ious]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    bars = ax.bar(np.arange(len(labels)), values, color="#DD8452")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("F1")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_collar_accuracy_curve(collar_results, out_path, title):
    """Accuracy vs collar width — shows how boundary ambiguity affects metrics."""
    collars = [r["collar_frames"] for r in collar_results]
    accs = [r["accuracy"] for r in collar_results]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.plot(collars, accs, marker="o", linewidth=2, color="#55A868")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Collar width (±frames excluded)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    # Add secondary x-axis for seconds (assuming 30fps)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(collars)
    ax2.set_xticklabels([f"±{c/30:.2f}s" for c in collars], fontsize=8)
    ax2.set_xlabel("Collar width (seconds at 30fps)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_transition_tolerance_curve(scores, out_path, title):
    tol = [s["tol_frames"] for s in scores]
    f1 = [s["f1"] for s in scores]
    prec = [s["precision"] for s in scores]
    rec = [s["recall"] for s in scores]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    ax.plot(tol, prec, marker="o", label="Precision")
    ax.plot(tol, rec, marker="o", label="Recall")
    ax.plot(tol, f1, marker="o", label="F1")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Tolerance (frames)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def bouts_to_duration_df(bouts, fps, session_prefix=None):
    rows = []
    for b in bouts:
        row = {
            "class_id": b.cls, "class": STATE_MAP.get(b.cls, str(b.cls)),
            "start_frame": b.start, "end_frame": b.end,
            "duration_frames": b.length,
            "duration_s": b.length / fps if fps > 0 else np.nan,
        }
        if session_prefix is not None:
            row["session_prefix"] = session_prefix
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bout_duration_distributions(gt_bouts, pr_bouts, fps, out_path, title,
                                     min_bouts_per_class=3):
    def summarize(bouts):
        by_cls = {c: [] for c in CLASSES}
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
    classes = [c for c in CLASSES if c in gt_sum and c in pr_sum]
    labels = [STATE_MAP[c] for c in classes]

    gt_med = [gt_sum[c][1] for c in classes]
    pr_med = [pr_sum[c][1] for c in classes]
    gt_err = [(gt_sum[c][1]-gt_sum[c][0], gt_sum[c][2]-gt_sum[c][1]) for c in classes]
    pr_err = [(pr_sum[c][1]-pr_sum[c][0], pr_sum[c][2]-pr_sum[c][1]) for c in classes]

    x = np.arange(len(classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    ax.bar(x - width/2, gt_med, width=width, label="GT (human)")
    ax.errorbar(x - width/2, gt_med,
                yerr=[[e[0] for e in gt_err], [e[1] for e in gt_err]],
                fmt="none", capsize=3, color="black")
    ax.bar(x + width/2, pr_med, width=width, label="Pred (model)")
    ax.errorbar(x + width/2, pr_med,
                yerr=[[e[0] for e in pr_err], [e[1] for e in pr_err]],
                fmt="none", capsize=3, color="black")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Bout duration (s), median ± IQR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_session_f1(all_metrics, out_path, title="Per-class F1 across Sessions"):
    sessions = sorted(all_metrics.keys())
    if not sessions:
        return
    n_sessions = len(sessions)
    labels = [STATE_MAP[c] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(max(9, n_sessions * 1.5), 5), dpi=200)
    width = 0.8 / n_sessions
    x = np.arange(N_CLASSES)

    for i, sess in enumerate(sessions):
        pc = all_metrics[sess].get("frame_metrics", {}).get("per_class", {})
        f1_vals = [pc.get(c, pc.get(str(c), {})).get("f1", 0.0) for c in CLASSES]
        offset = (i - n_sessions / 2 + 0.5) * width
        ax.bar(x + offset, f1_vals, width=width, label=sess)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("F1 (frame-level)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=min(n_sessions, 4))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_session_summary_bars(all_metrics, out_path, title="Session Summary"):
    sessions = sorted(all_metrics.keys())
    if not sessions:
        return
    accs = [all_metrics[s]["frame_metrics"]["accuracy"] for s in sessions]
    kappas = [all_metrics[s]["cohens_kappa"] for s in sessions]
    mf1s = [all_metrics[s]["frame_metrics"]["macro_f1"] for s in sessions]

    x = np.arange(len(sessions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(sessions) * 1.2), 4.5), dpi=200)
    ax.bar(x - width, accs, width=width, label="Accuracy")
    ax.bar(x, mf1s, width=width, label="Macro F1")
    ax.bar(x + width, kappas, width=width, label="Cohen's κ")
    ax.set_title(title, fontsize=12)
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
# Core evaluation
# ============================================================

def evaluate_arrays(gt, pr, ignore_label, iou_thr, tol_frames, fps,
                    collar_widths, session_prefix=None):
    """Compute all metrics for one gt/pred pair."""
    keep = gt != ignore_label
    gt_k = gt[keep]
    pr_k = pr[keep]

    # --- Frame-level ---
    fm = framewise_metrics(gt_k, pr_k)
    kappa = cohens_kappa(fm["confusion_matrix"])

    # --- Collar accuracy ---
    collar_results = [{"collar_frames": 0, "accuracy": fm["accuracy"],
                       "n_frames": int(len(gt_k))}]
    for cw in collar_widths:
        acc, n = collar_accuracy(gt_k, pr_k, cw)
        collar_results.append({"collar_frames": cw, "accuracy": acc, "n_frames": n})

    # --- Segment-level TAS metrics ---
    f1_at_ious = []
    for tau in [0.10, 0.25, 0.50]:
        f1_at_ious.append(segment_f1_at_iou(gt_k, pr_k, tau))

    edit = edit_score(gt_k, pr_k)

    # --- Bout-level (from bout_metrics.py) ---
    bm = bout_metrics(gt_k, pr_k, classes=CLASSES,
                       ignore_labels={ignore_label}, iou_thr=iou_thr)

    bt_scores_strict = []
    bt_scores_relaxed = []
    for tol in tol_frames:
        bt_scores_strict.append(boundary_tolerance_score(
            gt_k, pr_k, tol_frames=int(tol), ignore_labels={ignore_label},
            mode="strict"))
        bt_scores_relaxed.append(boundary_tolerance_score(
            gt_k, pr_k, tol_frames=int(tol), ignore_labels={ignore_label},
            mode="relaxed"))

    gt_bouts = labels_to_bouts(gt_k, ignore_labels={ignore_label})
    pr_bouts = labels_to_bouts(pr_k, ignore_labels={ignore_label})

    return {
        "gt_k": gt_k, "pr_k": pr_k,
        "frame_metrics": fm,
        "cohens_kappa": kappa,
        "collar_accuracy": collar_results,
        "segment_f1": f1_at_ious,
        "edit_score": edit,
        "bout_metrics": bm,
        "transition_tolerance_strict": bt_scores_strict,
        "transition_tolerance_relaxed": bt_scores_relaxed,
        "gt_bouts": gt_bouts, "pr_bouts": pr_bouts,
        "n_frames_total": int(len(gt)),
        "n_frames_used": int(len(gt_k)),
        "session_prefix": session_prefix,
    }


# ============================================================
# Console output
# ============================================================

def print_session_summary(result, prefix=""):
    """Print publication-style summary to console."""
    fm = result["frame_metrics"]
    kappa = result["cohens_kappa"]
    edit = result["edit_score"]
    f1_ious = result["segment_f1"]
    collar = result["collar_accuracy"]
    bm = result["bout_metrics"]

    tag = f"[Eval] {prefix}" if prefix else "[Eval]"
    print(f"\n{tag}")

    # Frame-level headline
    print(f"  Frame:  acc={fm['accuracy']:.3f}  macro_F1={fm['macro_f1']:.3f}  "
          f"weighted_F1={fm['weighted_f1']:.3f}  "
          f"Cohen's κ={kappa:.3f} ({kappa_interpretation(kappa)})")

    # Collar accuracy
    for cr in collar:
        if cr["collar_frames"] > 0:
            print(f"  Collar ±{cr['collar_frames']:d} frames: "
                  f"acc={cr['accuracy']:.3f} ({cr['n_frames']} frames)")

    # Segment-level TAS metrics
    f1_str = "  ".join(f"F1@{int(r['iou_threshold']*100)}={r['f1']:.3f}"
                       for r in f1_ious)
    print(f"  Segment: {f1_str}  Edit={edit:.1f}")

    # Per-class frame F1
    for c in CLASSES:
        d = fm["per_class"].get(c, {})
        print(f"    {STATE_MAP[c]:>10s}  P={d.get('precision',0):.2f}  "
              f"R={d.get('recall',0):.2f}  F1={d.get('f1',0):.2f}  "
              f"n={d.get('support',0)}")

    # Bout-level
    bov = bm.get("overall", {})
    if bov:
        print(f"  Bout:   P={bov.get('precision',0):.3f}  "
              f"R={bov.get('recall',0):.3f}  F1={bov.get('f1',0):.3f}  "
              f"mean_IoU={bov.get('mean_iou',0):.3f}")

    # Boundary tolerance
    for s in result["transition_tolerance_strict"]:
        print(f"  Boundary strict ±{s['tol_frames']}: "
              f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  "
              f"offset={s.get('mean_offset_frames',0):.1f}f")
    for s in result["transition_tolerance_relaxed"]:
        print(f"  Boundary relaxed ±{s['tol_frames']}: "
              f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  "
              f"offset={s.get('mean_offset_frames',0):.1f}f")


# ============================================================
# Write outputs for one session
# ============================================================

def write_session_outputs(result, out_dir, args, pred_csv_path, title_prefix=""):
    plots_dir = ensure_dir(out_dir / "plots")

    fm = result["frame_metrics"]
    kappa = result["cohens_kappa"]
    edit = result["edit_score"]
    f1_ious = result["segment_f1"]
    collar = result["collar_accuracy"]
    bm = result["bout_metrics"]
    bt_strict = result["transition_tolerance_strict"]
    bt_relaxed = result["transition_tolerance_relaxed"]
    gt_bouts = result["gt_bouts"]
    pr_bouts = result["pr_bouts"]

    # --- metrics.json ---
    metrics = {
        "input": {
            "pred_csv": str(Path(pred_csv_path).resolve()),
            "session_prefix": result.get("session_prefix"),
            "gt_col": args.gt_col, "pred_col": args.pred_col,
            "fps": args.fps, "ignore_label": args.ignore_label,
            "iou_thr": args.iou_thr,
            "n_frames_total": result["n_frames_total"],
            "n_frames_used": result["n_frames_used"],
        },
        "frame_metrics": {
            "accuracy": fm["accuracy"],
            "macro_f1": fm["macro_f1"],
            "weighted_f1": fm["weighted_f1"],
            "cohens_kappa": kappa,
            "kappa_interpretation": kappa_interpretation(kappa),
            "per_class": {STATE_MAP[c]: fm["per_class"][c] for c in CLASSES},
        },
        "collar_accuracy": collar,
        "segment_metrics": {
            "f1_at_ious": f1_ious,
            "edit_score": edit,
        },
        "bout_metrics": bm,
        "transition_tolerance_strict": bt_strict,
        "transition_tolerance_relaxed": bt_relaxed,
        "frame_confusion_counts": fm["confusion_matrix"].tolist(),
        "frame_confusion_normalized": normalize_rows(fm["confusion_matrix"]).tolist(),
    }
    save_json(metrics, out_dir / "metrics.json")

    # --- per_class.csv (frame-level + bout-level combined) ---
    rows = []
    for c in CLASSES:
        fd = fm["per_class"].get(c, {})
        bd = bm.get("per_class", {}).get(c, {})
        rows.append({
            "class_id": c, "class": STATE_MAP[c],
            "frame_precision": fd.get("precision", 0),
            "frame_recall": fd.get("recall", 0),
            "frame_f1": fd.get("f1", 0),
            "frame_support": fd.get("support", 0),
            "bout_precision": bd.get("precision", 0),
            "bout_recall": bd.get("recall", 0),
            "bout_f1": bd.get("f1", 0),
            "bout_gt": bd.get("gt_bouts", 0),
            "bout_pred": bd.get("pred_bouts", 0),
            "bout_mean_iou": bd.get("mean_iou", 0),
        })
    pd.DataFrame(rows).to_csv(out_dir / "per_class.csv", index=False)

    # --- summary.csv (one-row summary, easy to aggregate) ---
    summary = {
        "session": result.get("session_prefix", ""),
        "n_frames": result["n_frames_used"],
        "accuracy": fm["accuracy"],
        "macro_f1": fm["macro_f1"],
        "weighted_f1": fm["weighted_f1"],
        "cohens_kappa": kappa,
    }
    for cr in collar:
        if cr["collar_frames"] > 0:
            summary[f"collar_{cr['collar_frames']}_acc"] = cr["accuracy"]
    for r in f1_ious:
        summary[f"F1@{int(r['iou_threshold']*100)}"] = r["f1"]
    summary["edit_score"] = edit
    bov = bm.get("overall", {})
    summary["bout_f1"] = bov.get("f1", 0)
    summary["bout_mean_iou"] = bov.get("mean_iou", 0)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    # --- Plots ---
    tp = f"{title_prefix} — " if title_prefix else ""

    plot_confusion_matrix(
        fm["confusion_matrix"], CLASSES,
        plots_dir / "confusion_matrix.png",
        title=f"{tp}Frame Confusion Matrix")

    plot_per_class_f1(
        fm["per_class"], plots_dir / "frame_f1_per_class.png",
        title=f"{tp}Frame F1 by Class")

    if bm.get("per_class"):
        plot_per_class_f1(
            bm["per_class"], plots_dir / "bout_f1_per_class.png",
            title=f"{tp}Bout F1 by Class")

    plot_segment_f1_bars(
        f1_ious, plots_dir / "segment_f1_at_iou.png",
        title=f"{tp}Segment F1 @ IoU Thresholds")

    if len(collar) > 1:
        plot_collar_accuracy_curve(
            collar, plots_dir / "collar_accuracy_curve.png",
            title=f"{tp}Accuracy vs Boundary Collar Width")

    if bt_strict:
        plot_transition_tolerance_curve(
            bt_strict, plots_dir / "boundary_tolerance_strict.png",
            title=f"{tp}Boundary Tolerance — Strict (class-matched)")
    if bt_relaxed:
        plot_transition_tolerance_curve(
            bt_relaxed, plots_dir / "boundary_tolerance_relaxed.png",
            title=f"{tp}Boundary Tolerance — Relaxed (any transition)")

    plot_bout_duration_distributions(
        gt_bouts, pr_bouts, fps=args.fps,
        out_path=plots_dir / "bout_duration_comparison.png",
        title=f"{tp}Bout Duration (Median ± IQR)")

    # Bout tables
    bouts_to_duration_df(gt_bouts, args.fps, result.get("session_prefix")).to_csv(
        out_dir / "gt_bouts.csv", index=False)
    bouts_to_duration_df(pr_bouts, args.fps, result.get("session_prefix")).to_csv(
        out_dir / "pred_bouts.csv", index=False)

    # Console
    print_session_summary(result, title_prefix)


# ============================================================
# Discover prediction CSVs
# ============================================================

def discover_pred_csvs(pred_dir, session_prefix_filter=None):
    pred_dir = Path(pred_dir)
    csvs = sorted(pred_dir.glob("*_predictions.csv"))
    if not csvs:
        csvs = sorted(pred_dir.glob("*.csv"))

    results = []
    for csv_path in csvs:
        stem = csv_path.stem
        prefix = stem[:-len("_predictions")] if stem.endswith("_predictions") else stem
        if session_prefix_filter is not None:
            filt = session_prefix_filter.strip()
            if not (prefix == filt or prefix.startswith(filt)):
                continue
        results.append((prefix, str(csv_path)))

    if not results:
        raise FileNotFoundError(
            f"No prediction CSVs found in {pred_dir}"
            + (f" matching '{session_prefix_filter}'" if session_prefix_filter else ""))
    return results


# ============================================================
# Aggregate metrics across sessions
# ============================================================

def aggregate_results(all_results, args, out_dir):
    agg_dir = ensure_dir(out_dir / "aggregate")
    plots_dir = ensure_dir(agg_dir / "plots")

    gt_all = np.concatenate([r["gt_k"] for r in all_results.values()])
    pr_all = np.concatenate([r["pr_k"] for r in all_results.values()])

    # Frame-level aggregate
    fm = framewise_metrics(gt_all, pr_all)
    kappa = cohens_kappa(fm["confusion_matrix"])

    # Collar
    collar_results = [{"collar_frames": 0, "accuracy": fm["accuracy"],
                       "n_frames": int(len(gt_all))}]
    for cw in args.collar:
        acc, n = collar_accuracy(gt_all, pr_all, cw)
        collar_results.append({"collar_frames": cw, "accuracy": acc, "n_frames": n})

    # Segment-level (pooled)
    f1_at_ious = [segment_f1_at_iou(gt_all, pr_all, tau) for tau in [0.10, 0.25, 0.50]]
    edit = edit_score(gt_all, pr_all)

    # Bout-level
    bm = bout_metrics(gt_all, pr_all, classes=CLASSES,
                       ignore_labels={args.ignore_label}, iou_thr=args.iou_thr)
    bt_strict = [boundary_tolerance_score(gt_all, pr_all, tol_frames=int(tol),
                                           ignore_labels={args.ignore_label},
                                           mode="strict")
                 for tol in args.tol_frames]
    bt_relaxed = [boundary_tolerance_score(gt_all, pr_all, tol_frames=int(tol),
                                            ignore_labels={args.ignore_label},
                                            mode="relaxed")
                  for tol in args.tol_frames]
    gt_bouts = labels_to_bouts(gt_all, ignore_labels={args.ignore_label})
    pr_bouts = labels_to_bouts(pr_all, ignore_labels={args.ignore_label})

    sessions_list = sorted(all_results.keys())

    # --- Write aggregate metrics.json ---
    metrics = {
        "mode": "aggregate",
        "sessions": sessions_list, "n_sessions": len(sessions_list),
        "n_frames": int(len(gt_all)),
        "frame_metrics": {
            "accuracy": fm["accuracy"],
            "macro_f1": fm["macro_f1"],
            "weighted_f1": fm["weighted_f1"],
            "cohens_kappa": kappa,
            "kappa_interpretation": kappa_interpretation(kappa),
            "per_class": {STATE_MAP[c]: fm["per_class"][c] for c in CLASSES},
        },
        "collar_accuracy": collar_results,
        "segment_metrics": {"f1_at_ious": f1_at_ious, "edit_score": edit},
        "bout_metrics": bm,
        "transition_tolerance_strict": bt_strict,
        "transition_tolerance_relaxed": bt_relaxed,
        "per_session": {
            sess: {
                "accuracy": r["frame_metrics"]["accuracy"],
                "macro_f1": r["frame_metrics"]["macro_f1"],
                "cohens_kappa": r["cohens_kappa"],
                "F1@10": r["segment_f1"][0]["f1"],
                "F1@25": r["segment_f1"][1]["f1"],
                "F1@50": r["segment_f1"][2]["f1"],
                "edit_score": r["edit_score"],
            }
            for sess, r in all_results.items()
        },
    }
    save_json(metrics, agg_dir / "metrics.json")

    # --- Aggregate per_class.csv ---
    rows = []
    for c in CLASSES:
        fd = fm["per_class"].get(c, {})
        bd = bm.get("per_class", {}).get(c, {})
        rows.append({
            "class_id": c, "class": STATE_MAP[c],
            "frame_precision": fd.get("precision", 0),
            "frame_recall": fd.get("recall", 0),
            "frame_f1": fd.get("f1", 0),
            "frame_support": fd.get("support", 0),
            "bout_f1": bd.get("f1", 0),
            "bout_mean_iou": bd.get("mean_iou", 0),
        })
    pd.DataFrame(rows).to_csv(agg_dir / "per_class.csv", index=False)

    # --- Session summary table ---
    summary_rows = []
    for sess in sessions_list:
        r = all_results[sess]
        row = {
            "session": sess,
            "n_frames": r["n_frames_used"],
            "accuracy": r["frame_metrics"]["accuracy"],
            "macro_f1": r["frame_metrics"]["macro_f1"],
            "weighted_f1": r["frame_metrics"]["weighted_f1"],
            "cohens_kappa": r["cohens_kappa"],
        }
        for cr in r["collar_accuracy"]:
            if cr["collar_frames"] > 0:
                row[f"collar_{cr['collar_frames']}_acc"] = cr["accuracy"]
        for ri in r["segment_f1"]:
            row[f"F1@{int(ri['iou_threshold']*100)}"] = ri["f1"]
        row["edit_score"] = r["edit_score"]
        bov = r["bout_metrics"].get("overall", {})
        row["bout_f1"] = bov.get("f1", 0)
        summary_rows.append(row)

    # Add aggregate row
    agg_row = {"session": "AGGREGATE", "n_frames": int(len(gt_all)),
               "accuracy": fm["accuracy"], "macro_f1": fm["macro_f1"],
               "weighted_f1": fm["weighted_f1"], "cohens_kappa": kappa}
    for cr in collar_results:
        if cr["collar_frames"] > 0:
            agg_row[f"collar_{cr['collar_frames']}_acc"] = cr["accuracy"]
    for ri in f1_at_ious:
        agg_row[f"F1@{int(ri['iou_threshold']*100)}"] = ri["f1"]
    agg_row["edit_score"] = edit
    bov = bm.get("overall", {})
    agg_row["bout_f1"] = bov.get("f1", 0)
    summary_rows.append(agg_row)

    pd.DataFrame(summary_rows).to_csv(agg_dir / "session_summary.csv", index=False)

    # --- Plots ---
    plot_confusion_matrix(fm["confusion_matrix"], CLASSES,
                          plots_dir / "confusion_matrix.png",
                          title="Aggregate — Frame Confusion Matrix")
    plot_per_class_f1(fm["per_class"], plots_dir / "frame_f1_per_class.png",
                      title="Aggregate — Frame F1 by Class")
    plot_segment_f1_bars(f1_at_ious, plots_dir / "segment_f1_at_iou.png",
                         title="Aggregate — Segment F1 @ IoU")
    if len(collar_results) > 1:
        plot_collar_accuracy_curve(collar_results,
                                   plots_dir / "collar_accuracy_curve.png",
                                   title="Aggregate — Collar Accuracy")
    if bt_strict:
        plot_transition_tolerance_curve(
            bt_strict, plots_dir / "boundary_tolerance_strict.png",
            title="Aggregate — Boundary Tolerance (strict)")
    if bt_relaxed:
        plot_transition_tolerance_curve(
            bt_relaxed, plots_dir / "boundary_tolerance_relaxed.png",
            title="Aggregate — Boundary Tolerance (relaxed)")
    plot_bout_duration_distributions(gt_bouts, pr_bouts, fps=args.fps,
                                     out_path=plots_dir / "bout_duration.png",
                                     title="Aggregate — Bout Duration")

    # Cross-session comparison
    plot_cross_session_f1(all_results, plots_dir / "cross_session_f1.png")
    plot_session_summary_bars(all_results, plots_dir / "session_summary.png")

    # Bout tables
    bouts_to_duration_df(gt_bouts, args.fps).to_csv(agg_dir / "gt_bouts.csv", index=False)
    bouts_to_duration_df(pr_bouts, args.fps).to_csv(agg_dir / "pred_bouts.csv", index=False)

    # Console
    print(f"\n{'='*60}")
    print(f"[Aggregate] {len(sessions_list)} sessions, {len(gt_all)} frames")
    print(f"  Frame:  acc={fm['accuracy']:.3f}  macro_F1={fm['macro_f1']:.3f}  "
          f"Cohen's κ={kappa:.3f} ({kappa_interpretation(kappa)})")
    for cr in collar_results:
        if cr["collar_frames"] > 0:
            print(f"  Collar ±{cr['collar_frames']}: acc={cr['accuracy']:.3f}")
    f1_str = "  ".join(f"F1@{int(r['iou_threshold']*100)}={r['f1']:.3f}" for r in f1_at_ious)
    print(f"  Segment: {f1_str}  Edit={edit:.1f}")
    bov = bm.get("overall", {})
    print(f"  Bout:   P={bov.get('precision',0):.3f}  R={bov.get('recall',0):.3f}  "
          f"F1={bov.get('f1',0):.3f}")
    print(f"[Aggregate] Wrote: {agg_dir}")


# ============================================================
# Majority vote comparison
# ============================================================

def vote_comparison_single(gt, pr, vote_windows, ignore_label, iou_thr,
                           tol_frames, fps, collar_widths, session_prefix, out_dir):
    """Run metrics for raw + each vote window, print table, save CSV + plot."""
    rows = []

    # Raw (no vote)
    result_raw = evaluate_arrays(gt, pr, ignore_label, iou_thr, tol_frames,
                                 fps, collar_widths, session_prefix)
    fm = result_raw["frame_metrics"]
    rows.append({
        "smoothing": "raw",
        "window": 0,
        "accuracy": fm["accuracy"],
        "macro_f1": fm["macro_f1"],
        "weighted_f1": fm["weighted_f1"],
        "cohens_kappa": result_raw["cohens_kappa"],
        "F1@10": result_raw["segment_f1"][0]["f1"],
        "F1@25": result_raw["segment_f1"][1]["f1"],
        "F1@50": result_raw["segment_f1"][2]["f1"],
        "edit_score": result_raw["edit_score"],
        "bout_f1": result_raw["bout_metrics"].get("overall", {}).get("f1", 0),
    })

    for win in vote_windows:
        pr_voted = majority_vote(pr, win, ignore_label)
        result_v = evaluate_arrays(gt, pr_voted, ignore_label, iou_thr, tol_frames,
                                   fps, collar_widths, session_prefix)
        fmv = result_v["frame_metrics"]
        rows.append({
            "smoothing": f"vote_{win}",
            "window": win,
            "accuracy": fmv["accuracy"],
            "macro_f1": fmv["macro_f1"],
            "weighted_f1": fmv["weighted_f1"],
            "cohens_kappa": result_v["cohens_kappa"],
            "F1@10": result_v["segment_f1"][0]["f1"],
            "F1@25": result_v["segment_f1"][1]["f1"],
            "F1@50": result_v["segment_f1"][2]["f1"],
            "edit_score": result_v["edit_score"],
            "bout_f1": result_v["bout_metrics"].get("overall", {}).get("f1", 0),
        })

    df = pd.DataFrame(rows)
    csv_path = Path(out_dir) / "vote_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Console table
    tag = f" ({session_prefix})" if session_prefix else ""
    print(f"\n{'='*80}")
    print(f"  Majority Vote Comparison{tag}")
    print(f"{'='*80}")
    print(f"  {'Window':>8s}  {'Acc':>6s}  {'mF1':>6s}  {'wF1':>6s}  "
          f"{'κ':>6s}  {'F1@10':>6s}  {'F1@25':>6s}  {'F1@50':>6s}  "
          f"{'Edit':>5s}  {'Bout':>6s}")
    print(f"  {'-'*74}")
    for _, r in df.iterrows():
        lbl = "raw" if r["window"] == 0 else f"±{r['window']//2}f"
        print(f"  {lbl:>8s}  {r['accuracy']:.3f}  {r['macro_f1']:.3f}  "
              f"{r['weighted_f1']:.3f}  {r['cohens_kappa']:.3f}  "
              f"{r['F1@10']:.3f}  {r['F1@25']:.3f}  {r['F1@50']:.3f}  "
              f"{r['edit_score']:5.1f}  {r['bout_f1']:.3f}")
    # Highlight best
    best_idx = df["macro_f1"].idxmax()
    best = df.iloc[best_idx]
    best_lbl = "raw" if best["window"] == 0 else f"vote_{int(best['window'])}"
    print(f"  {'':>8s}  → Best macro_F1: {best['macro_f1']:.3f} ({best_lbl})")
    print(f"{'='*80}")

    # Plot
    plot_vote_comparison(df, Path(out_dir) / "plots" / "vote_comparison.png",
                         title=f"Majority Vote Smoothing{tag}")
    print(f"  [Vote] Saved: {csv_path}")
    return df


def vote_comparison_batch(all_results, vote_windows, args, out_dir):
    """Aggregate all sessions, then run vote comparison on pooled predictions."""
    gt_all = np.concatenate([r["gt_k"] for r in all_results.values()])
    pr_all = np.concatenate([r["pr_k"] for r in all_results.values()])
    vote_comparison_single(gt_all, pr_all, vote_windows, args.ignore_label,
                           args.iou_thr, args.tol_frames, args.fps,
                           args.collar, "aggregate", out_dir / "aggregate")


def plot_vote_comparison(df, out_path, title=""):
    """Bar chart comparing key metrics across vote windows."""
    ensure_dir(out_path.parent)
    metrics = ["accuracy", "macro_f1", "cohens_kappa", "F1@25", "bout_f1"]
    labels = [r["smoothing"] for _, r in df.iterrows()]
    x = np.arange(len(labels))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5), dpi=200)
    for i, m in enumerate(metrics):
        vals = df[m].values
        bars = ax.bar(x + i * width - (n_metrics - 1) * width / 2, vals,
                      width=width, label=m, alpha=0.85)
        # Add value labels on top
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, df[metrics].values.max() * 1.15))
    ax.set_title(title or "Majority Vote Comparison", fontsize=12)
    ax.legend(fontsize=8, frameon=False, ncol=n_metrics)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate TCN behavior predictions with publication-standard metrics.")

    ap.add_argument("--pred_csv", default=None)
    ap.add_argument("--pred_dir", default=None)
    ap.add_argument("--session_prefix", default=None)
    ap.add_argument("--gt_col", default="human_labeled_state")
    ap.add_argument("--pred_col", default="tcn_final")
    ap.add_argument("--out_dir", default="outputs/metrics")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ignore_label", type=int, default=-1)
    ap.add_argument("--iou_thr", type=float, default=0.1,
                    help="IoU threshold for bout matching.")
    ap.add_argument("--tol_frames", type=int, nargs="+", default=[3, 5, 10, 15],
                    help="Boundary tolerance windows (frames).")
    ap.add_argument("--collar", type=int, nargs="+", default=[5, 10, 15],
                    help="Collar widths for collar accuracy "
                         "(±N frames around GT transitions excluded). "
                         "Default: 5 10 15 (= ±0.17s, ±0.33s, ±0.50s at 30fps)")
    ap.add_argument("--vote", type=int, nargs="*", default=None,
                    help="Majority vote smoothing windows (frames). "
                         "Evaluates raw + each window for comparison. "
                         "Example: --vote 10 20 30")
    ap.add_argument("--enforce_still", action="store_true",
                    help="Correct GT labels to 'still' where speed stays below "
                         "--still_speed_thr for > --still_min_frames consecutive frames.")
    ap.add_argument("--speed_col", default="speed_cm_per_s",
                    help="Column name for speed in prediction CSV (default: speed_cm_per_s).")
    ap.add_argument("--still_speed_thr", type=float, default=0.1,
                    help="Speed threshold in cm/s for stillness enforcement (default: 0.1).")
    ap.add_argument("--still_min_frames", type=int, default=30,
                    help="Minimum consecutive low-speed frames to override to 'still' "
                         "(default: 30 = 1s at 30fps).")

    args = ap.parse_args()

    if args.pred_csv is None and args.pred_dir is None:
        raise ValueError("Provide --pred_csv or --pred_dir.")

    out_dir = ensure_dir(args.out_dir)

    # ============================
    # Single session
    # ============================
    if args.pred_csv is not None and args.pred_dir is None:
        print(f"[Mode] Single CSV: {args.pred_csv}")
        df = pd.read_csv(args.pred_csv)
        assert args.gt_col in df.columns, f"Missing '{args.gt_col}' in {list(df.columns)}"
        assert args.pred_col in df.columns, f"Missing '{args.pred_col}' in {list(df.columns)}"

        gt = df[args.gt_col].to_numpy(dtype=int)
        pr = df[args.pred_col].to_numpy(dtype=int)

        # --- Enforce stillness by speed (optional GT correction) ---
        if args.enforce_still:
            if args.speed_col in df.columns:
                speed = df[args.speed_col].to_numpy(dtype=float)
                gt = enforce_stillness_by_speed(
                    gt, speed,
                    threshold=args.still_speed_thr,
                    min_duration=args.still_min_frames)
            else:
                print(f"  [WARN] --enforce_still requested but '{args.speed_col}' "
                      f"not found in CSV columns: {list(df.columns)}")

        prefix = args.session_prefix
        if prefix is None and "session_prefix" in df.columns:
            prefix = str(df["session_prefix"].iloc[0])

        result = evaluate_arrays(
            gt, pr, ignore_label=args.ignore_label, iou_thr=args.iou_thr,
            tol_frames=args.tol_frames, fps=args.fps, collar_widths=args.collar,
            session_prefix=prefix)

        title_prefix = prefix or Path(args.pred_csv).stem
        write_session_outputs(result, out_dir, args, args.pred_csv, title_prefix)

        # --- Vote comparison ---
        if args.vote:
            vote_comparison_single(gt, pr, args.vote, args.ignore_label,
                                   args.iou_thr, args.tol_frames, args.fps,
                                   args.collar, prefix, out_dir)

        print(f"\n[OK] Outputs in {out_dir}")

    # ============================
    # Batch mode
    # ============================
    else:
        pred_dir = args.pred_dir or str(Path(args.pred_csv).parent)
        print(f"[Mode] Batch from: {pred_dir}")
        session_csvs = discover_pred_csvs(pred_dir, args.session_prefix)

        print(f"[Found] {len(session_csvs)} session(s):")
        for prefix, csv_path in session_csvs:
            print(f"  - {prefix}: {csv_path}")

        all_results = {}
        errors = []

        for prefix, csv_path in session_csvs:
            try:
                df = pd.read_csv(csv_path)
                assert args.gt_col in df.columns, f"Missing '{args.gt_col}'"
                assert args.pred_col in df.columns, f"Missing '{args.pred_col}'"

                gt = df[args.gt_col].to_numpy(dtype=int)
                pr = df[args.pred_col].to_numpy(dtype=int)

                # --- Enforce stillness by speed (optional GT correction) ---
                if args.enforce_still:
                    if args.speed_col in df.columns:
                        speed = df[args.speed_col].to_numpy(dtype=float)
                        gt = enforce_stillness_by_speed(
                            gt, speed,
                            threshold=args.still_speed_thr,
                            min_duration=args.still_min_frames)
                    else:
                        print(f"  [WARN] --enforce_still: '{args.speed_col}' "
                              f"not in {prefix} CSV columns")

                result = evaluate_arrays(
                    gt, pr, ignore_label=args.ignore_label, iou_thr=args.iou_thr,
                    tol_frames=args.tol_frames, fps=args.fps,
                    collar_widths=args.collar, session_prefix=prefix)

                sess_dir = ensure_dir(out_dir / prefix)
                write_session_outputs(result, sess_dir, args, csv_path, prefix)
                all_results[prefix] = result

            except Exception as e:
                print(f"  [ERROR] {prefix}: {e}")
                errors.append((prefix, str(e)))

        if len(all_results) > 1:
            aggregate_results(all_results, args, out_dir)

        # --- Vote comparison (aggregate) ---
        if args.vote and len(all_results) > 0:
            if len(all_results) > 1:
                vote_comparison_batch(all_results, args.vote, args, out_dir)
            else:
                # Single session in batch mode
                prefix, res = next(iter(all_results.items()))
                vote_comparison_single(
                    res["gt_k"], res["pr_k"], args.vote, args.ignore_label,
                    args.iou_thr, args.tol_frames, args.fps,
                    args.collar, prefix, out_dir / prefix)

        print(f"\n{'='*60}")
        print(f"[Done] {len(all_results)}/{len(session_csvs)} sessions evaluated.")
        if errors:
            print(f"[Errors] {len(errors)} failed:")
            for prefix, err in errors:
                print(f"  - {prefix}: {err}")
        print(f"[Outputs] {out_dir}")


if __name__ == "__main__":
    main()