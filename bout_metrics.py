"""
bout_metrics.py — Publication-standard bout and segment evaluation metrics

Metrics follow conventions from:
  - Temporal action segmentation (TAS): F1@{10,25,50}, Edit Score
    (Lea et al. 2017; Farha & Gall 2019)
  - Behavioral neuroscience (SimBA, MARS, B-SOiD): bout P/R/F1, Cohen's kappa
  - Standard boundary handling: collar exclusion (±K frames around transitions)

Key improvements over previous version:
  - Boundary tolerance has TWO modes:
      strict:  requires (from_class → to_class) match (original behavior)
      relaxed: requires only that a transition exists within ±tol (class-agnostic)
    Relaxed mode answers "did the model detect that *something* changed?"
  - Collar-aware bout metrics: option to exclude ±K frames around GT transitions
    before computing bout IoU, so boundary disagreement doesn't penalize bouts
  - Segment-level F1@τ and Edit Score built in (also available in evaluate.py)
  - Per-class and overall duration statistics
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set


# ============================================================
# Bout dataclass
# ============================================================

@dataclass
class Bout:
    cls: int
    start: int   # inclusive
    end: int     # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


# ============================================================
# Label ↔ bout conversions
# ============================================================

def labels_to_bouts(labels: np.ndarray, ignore_labels: Set[int] = {-1}) -> List[Bout]:
    """Convert frame-level labels to list of contiguous Bout objects."""
    labels = np.asarray(labels, dtype=int)
    bouts: List[Bout] = []
    T = len(labels)
    i = 0
    while i < T:
        cls = int(labels[i])
        j = i + 1
        while j < T and int(labels[j]) == cls:
            j += 1
        if cls not in ignore_labels:
            bouts.append(Bout(cls=cls, start=i, end=j))
        i = j
    return bouts


def bouts_to_labels(bouts: List[Bout], T: int, default: int = -1) -> np.ndarray:
    """Convert list of Bouts back to frame-level label array."""
    labels = np.full(T, default, dtype=int)
    for b in bouts:
        labels[b.start:b.end] = b.cls
    return labels


# ============================================================
# Bout IoU
# ============================================================

def bout_iou(a: Bout, b: Bout) -> float:
    """Standard temporal IoU between two bouts."""
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter == 0:
        return 0.0
    union = a.length + b.length - inter
    return inter / union if union > 0 else 0.0


def bout_iou_with_collar(a: Bout, b: Bout, collar_mask: np.ndarray = None) -> float:
    """
    IoU computed only on frames OUTSIDE the collar zone.
    Collar-masked frames (around GT transitions) are excluded from both
    intersection and union, so boundary disagreement doesn't penalize.
    Falls back to standard IoU if no collar mask is provided.
    """
    if collar_mask is None:
        return bout_iou(a, b)

    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if lo >= hi:
        return 0.0

    # Count valid (non-collar) frames in intersection and each bout
    inter_valid = collar_mask[lo:hi].sum()
    a_valid = collar_mask[a.start:a.end].sum()
    b_valid = collar_mask[b.start:b.end].sum()

    union_valid = a_valid + b_valid - inter_valid
    if union_valid <= 0:
        return 0.0
    return float(inter_valid / union_valid)


# ============================================================
# Greedy bout matching
# ============================================================

def greedy_match_bouts(
    gt: List[Bout],
    pred: List[Bout],
    iou_thr: float = 0.1,
    collar_mask: np.ndarray = None,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match predicted bouts to GT bouts of the SAME CLASS using greedy IoU.

    Args:
        gt: ground-truth bouts
        pred: predicted bouts
        iou_thr: minimum IoU for a valid match
        collar_mask: optional boolean mask (True = valid frame, False = collar zone)

    Returns:
        matches: list of (gt_idx, pred_idx, iou)
        unmatched_gt: list of gt indices
        unmatched_pred: list of pred indices
    """
    iou_fn = bout_iou if collar_mask is None else \
        lambda a, b: bout_iou_with_collar(a, b, collar_mask)

    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.cls != p.cls:
                continue
            iou = iou_fn(g, p)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    gt_used: Set[int] = set()
    pred_used: Set[int] = set()
    matches = []
    for iou, gi, pi in candidates:
        if gi in gt_used or pi in pred_used:
            continue
        gt_used.add(gi)
        pred_used.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt)) if i not in gt_used]
    unmatched_pred = [i for i in range(len(pred)) if i not in pred_used]
    return matches, unmatched_gt, unmatched_pred


# ============================================================
# Core bout-level metrics
# ============================================================

def bout_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    classes: List[int] = [0, 1, 2, 3, 4, 5],
    ignore_labels: Set[int] = {-1},
    iou_thr: float = 0.1,
    collar_frames: int = 0,
) -> Dict:
    """
    Bout-level event detection metrics with optional collar.

    Args:
        collar_frames: if > 0, exclude ±collar_frames around GT transitions
                       when computing IoU. This prevents boundary disagreement
                       from counting as missed/false bouts.
    """
    gt_bouts_all = labels_to_bouts(gt_labels, ignore_labels=ignore_labels)
    pr_bouts_all = labels_to_bouts(pred_labels, ignore_labels=ignore_labels)

    # Build collar mask if requested
    collar_mask = None
    if collar_frames > 0:
        T = max(len(gt_labels), len(pred_labels))
        collar_mask = _build_collar_mask(gt_labels, collar_frames, T)

    # Overall matching
    matches, un_gt, un_pr = greedy_match_bouts(
        gt_bouts_all, pr_bouts_all, iou_thr=iou_thr, collar_mask=collar_mask)

    TP = len(matches)
    FP = len(un_pr)
    FN = len(un_gt)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    ious = np.array([m[2] for m in matches], dtype=float)
    mean_iou = float(ious.mean()) if len(ious) else 0.0

    # Duration-weighted IoU
    if len(matches):
        weights = np.array([gt_bouts_all[gi].length for gi, _, _ in matches], dtype=float)
        wmean_iou = float((ious * weights).sum() / weights.sum()) if weights.sum() else 0.0
    else:
        wmean_iou = 0.0

    # Per-class metrics
    per_class = {}
    for c in classes:
        gt_c = [b for b in gt_bouts_all if b.cls == c]
        pr_c = [b for b in pr_bouts_all if b.cls == c]
        m_c, un_gt_c, un_pr_c = greedy_match_bouts(
            gt_c, pr_c, iou_thr=iou_thr, collar_mask=collar_mask)
        TPc = len(m_c)
        FPc = len(un_pr_c)
        FNc = len(un_gt_c)
        pc = TPc / (TPc + FPc) if (TPc + FPc) else 0.0
        rc = TPc / (TPc + FNc) if (TPc + FNc) else 0.0
        f1c = (2 * pc * rc) / (pc + rc) if (pc + rc) else 0.0

        ious_c = np.array([x[2] for x in m_c], dtype=float)
        miou_c = float(ious_c.mean()) if len(ious_c) else 0.0

        # Duration statistics
        gt_durs = np.array([b.length for b in gt_c], dtype=float)
        pr_durs = np.array([b.length for b in pr_c], dtype=float)

        per_class[c] = {
            "TP": TPc, "FP": FPc, "FN": FNc,
            "precision": pc, "recall": rc, "f1": f1c,
            "mean_iou": miou_c,
            "gt_bouts": len(gt_c), "pred_bouts": len(pr_c),
            "gt_dur_median": float(np.median(gt_durs)) if len(gt_durs) else 0.0,
            "gt_dur_iqr": float(np.subtract(*np.percentile(gt_durs, [75, 25]))) if len(gt_durs) >= 2 else 0.0,
            "pred_dur_median": float(np.median(pr_durs)) if len(pr_durs) else 0.0,
            "pred_dur_iqr": float(np.subtract(*np.percentile(pr_durs, [75, 25]))) if len(pr_durs) >= 2 else 0.0,
        }

    return {
        "overall": {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_iou": mean_iou,
            "weighted_mean_iou": wmean_iou,
            "gt_bouts": len(gt_bouts_all),
            "pred_bouts": len(pr_bouts_all),
            "iou_thr": iou_thr,
            "collar_frames": collar_frames,
        },
        "per_class": per_class,
    }


# ============================================================
# Transition / boundary detection
# ============================================================

def _build_collar_mask(labels: np.ndarray, collar_frames: int, T: int = None) -> np.ndarray:
    """Boolean mask: True for frames OUTSIDE ±collar_frames of any GT transition."""
    if T is None:
        T = len(labels)
    mask = np.ones(T, dtype=bool)
    transitions = np.where(labels[1:] != labels[:-1])[0] + 1
    for t in transitions:
        lo = max(0, t - collar_frames)
        hi = min(T, t + collar_frames + 1)
        mask[lo:hi] = False
    return mask


def transition_times(labels: np.ndarray,
                     ignore_labels: Set[int] = {-1}) -> List[Tuple[int, int, int]]:
    """
    Returns list of transitions: (frame, from_cls, to_cls).
    Transition occurs at frame t where labels[t-1] → labels[t].
    """
    labels = np.asarray(labels, dtype=int)
    out = []
    for t in range(1, len(labels)):
        a, b = int(labels[t - 1]), int(labels[t])
        if a == b:
            continue
        if a in ignore_labels or b in ignore_labels:
            continue
        out.append((t, a, b))
    return out


def boundary_tolerance_score(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    tol_frames: int = 3,
    ignore_labels: Set[int] = {-1},
    mode: str = "strict",
) -> Dict:
    """
    Boundary detection with temporal tolerance.

    Two modes:
      strict  (original): transition must match both (from_class → to_class)
              and be within ±tol_frames. Use for evaluating whether the model
              captures the exact behavioral transition type.

      relaxed (new): any GT transition is matched if ANY predicted transition
              occurs within ±tol_frames, regardless of class labels.
              Use for evaluating "does the model detect that behavior changed?"
              This is more forgiving and more relevant for downstream analysis
              where knowing *when* behavior changed matters more than the
              exact from/to identity.

    Args:
        mode: "strict" or "relaxed"
    """
    gt_tr = transition_times(gt_labels, ignore_labels=ignore_labels)
    pr_tr = transition_times(pred_labels, ignore_labels=ignore_labels)

    pr_used: Set[int] = set()
    correct = 0
    offsets = []  # temporal offset of matched transitions

    for (t, a, b) in gt_tr:
        candidates = []
        for j, (tp, ap, bp) in enumerate(pr_tr):
            if j in pr_used:
                continue
            if abs(tp - t) > tol_frames:
                continue
            if mode == "strict" and (ap != a or bp != b):
                continue
            # mode == "relaxed": any transition within tolerance counts
            candidates.append((abs(tp - t), j))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            offset, jbest = candidates[0]
            pr_used.add(jbest)
            correct += 1
            offsets.append(offset)

    total_gt = len(gt_tr)
    total_pr = len(pr_tr)
    recall = correct / total_gt if total_gt else 0.0
    precision = correct / total_pr if total_pr else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    mean_offset = float(np.mean(offsets)) if offsets else 0.0

    return {
        "tol_frames": tol_frames,
        "mode": mode,
        "gt_transitions": total_gt,
        "pred_transitions": total_pr,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_offset_frames": mean_offset,
    }


# ============================================================
# Segment-level TAS metrics
# ============================================================

def labels_to_segments(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Convert labels to list of (class, start, end) segments."""
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


def segment_iou(seg_a: Tuple[int, int, int], seg_b: Tuple[int, int, int]) -> float:
    """IoU between two (class, start, end) segments."""
    _, s1, e1 = seg_a
    _, s2, e2 = seg_b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def segment_f1_at_iou(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    iou_threshold: float,
    classes: List[int] = [0, 1, 2, 3, 4, 5],
) -> Dict:
    """
    Segment-level F1 at a given IoU threshold (F1@τ).
    Standard TAS metric (Lea et al. 2017, Farha & Gall 2019).
    """
    gt_segs = labels_to_segments(gt_labels)
    pred_segs = labels_to_segments(pred_labels)

    tp_total, fp_total, fn_total = 0, 0, 0

    for c in classes:
        gt_c = [s for s in gt_segs if s[0] == c]
        pred_c = [s for s in pred_segs if s[0] == c]

        matched_gt: Set[int] = set()
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


def edit_score(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """
    Normalized Edit Score (Lea et al. 2017).
    Levenshtein distance on ordered segment class sequence.
    Score = (1 - edit_distance / max(|gt_segs|, |pred_segs|)) * 100
    Higher is better. 100 = perfect segment ordering.
    """
    gt_seq = [s[0] for s in labels_to_segments(gt_labels)]
    pred_seq = [s[0] for s in labels_to_segments(pred_labels)]

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
    return float(max(0.0, (1.0 - dist / max(n, m)) * 100.0))


# ============================================================
# Convenience: compute all standard metrics at once
# ============================================================

def compute_all_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    classes: List[int] = [0, 1, 2, 3, 4, 5],
    ignore_labels: Set[int] = {-1},
    iou_thr: float = 0.1,
    collar_frames: int = 0,
    tol_frames_list: List[int] = [3, 5, 10, 15],
    fps: float = 30.0,
) -> Dict:
    """
    One-call function that computes all publication-standard metrics.

    Returns dict with:
      bout_metrics: overall + per-class bout P/R/F1/IoU
      segment_f1: F1@{10, 25, 50}
      edit_score: normalized Levenshtein on segment ordering
      boundary_strict: boundary P/R/F1 at each tolerance (exact class match)
      boundary_relaxed: boundary P/R/F1 at each tolerance (class-agnostic)
    """
    # Filter ignored frames
    keep = ~np.isin(gt_labels, list(ignore_labels))
    gt = gt_labels[keep]
    pr = pred_labels[keep]

    # Bout-level
    bm = bout_metrics(gt, pr, classes=classes, ignore_labels=ignore_labels,
                       iou_thr=iou_thr, collar_frames=collar_frames)

    # Segment F1@τ
    seg_f1 = {}
    for tau in [0.10, 0.25, 0.50]:
        result = segment_f1_at_iou(gt, pr, tau, classes=classes)
        seg_f1[f"F1@{int(tau*100)}"] = result

    # Edit score
    es = edit_score(gt, pr)

    # Boundary tolerance (both modes)
    boundary_strict = []
    boundary_relaxed = []
    for tol in tol_frames_list:
        boundary_strict.append(
            boundary_tolerance_score(gt, pr, tol_frames=tol,
                                     ignore_labels=ignore_labels, mode="strict"))
        boundary_relaxed.append(
            boundary_tolerance_score(gt, pr, tol_frames=tol,
                                     ignore_labels=ignore_labels, mode="relaxed"))

    return {
        "bout_metrics": bm,
        "segment_f1": seg_f1,
        "edit_score": es,
        "boundary_strict": boundary_strict,
        "boundary_relaxed": boundary_relaxed,
    }