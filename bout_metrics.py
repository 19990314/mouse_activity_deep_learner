import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Bout:
    cls: int
    start: int  # inclusive
    end: int    # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start

def labels_to_bouts(labels: np.ndarray, ignore_labels: set = {-1}) -> List[Bout]:
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

def bout_iou(a: Bout, b: Bout) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter == 0:
        return 0.0
    union = (a.length + b.length - inter)
    return inter / union if union > 0 else 0.0

def greedy_match_bouts(
    gt: List[Bout],
    pred: List[Bout],
    iou_thr: float = 0.1,
) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
    """
    Match bouts of SAME CLASS using greedy IoU ranking.
    Returns:
      matches: list of (gt_idx, pred_idx, iou)
      unmatched_gt: list of gt indices
      unmatched_pred: list of pred indices
    """
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.cls != p.cls:
                continue
            iou = bout_iou(g, p)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    gt_used = set()
    pred_used = set()
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

def bout_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    classes: List[int] = [0,1,2,3,4,5],
    ignore_labels: set = {-1},
    iou_thr: float = 0.1,
) -> Dict:
    """
    Computes bout-level event detection metrics and IoU summaries.
    """
    gt_bouts_all = labels_to_bouts(gt_labels, ignore_labels=ignore_labels)
    pr_bouts_all = labels_to_bouts(pred_labels, ignore_labels=ignore_labels)

    # Overall matching across all classes (still class-consistent)
    matches, un_gt, un_pr = greedy_match_bouts(gt_bouts_all, pr_bouts_all, iou_thr=iou_thr)

    TP = len(matches)
    FP = len(un_pr)
    FN = len(un_gt)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

    # IoU summaries for matched bouts
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
        m_c, un_gt_c, un_pr_c = greedy_match_bouts(gt_c, pr_c, iou_thr=iou_thr)
        TPc = len(m_c)
        FPc = len(un_pr_c)
        FNc = len(un_gt_c)
        pc = TPc / (TPc + FPc) if (TPc + FPc) else 0.0
        rc = TPc / (TPc + FNc) if (TPc + FNc) else 0.0
        f1c = (2*pc*rc)/(pc+rc) if (pc+rc) else 0.0

        ious_c = np.array([x[2] for x in m_c], dtype=float)
        miou_c = float(ious_c.mean()) if len(ious_c) else 0.0

        per_class[c] = {
            "TP": TPc, "FP": FPc, "FN": FNc,
            "precision": pc, "recall": rc, "f1": f1c,
            "mean_iou": miou_c,
            "gt_bouts": len(gt_c),
            "pred_bouts": len(pr_c),
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
        },
        "per_class": per_class,
    }

def transition_times(labels: np.ndarray, ignore_labels: set = {-1}) -> List[Tuple[int,int,int]]:
    """
    Returns list of transitions: (t, from_cls, to_cls) where transition occurs at t (i.e., labels[t-1]->labels[t]).
    """
    labels = np.asarray(labels, dtype=int)
    out = []
    for t in range(1, len(labels)):
        a, b = int(labels[t-1]), int(labels[t])
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
    ignore_labels: set = {-1},
) -> Dict:
    """
    Matches transitions (from->to) by time with tolerance.
    A GT transition is correct if there exists an unmatched predicted transition of same (from,to)
    within +/- tol_frames.
    """
    gt_tr = transition_times(gt_labels, ignore_labels=ignore_labels)
    pr_tr = transition_times(pred_labels, ignore_labels=ignore_labels)

    pr_used = set()
    correct = 0

    for (t, a, b) in gt_tr:
        # find any pred transition matching (a,b) within tolerance
        candidates = []
        for j, (tp, ap, bp) in enumerate(pr_tr):
            if j in pr_used:
                continue
            if ap == a and bp == b and abs(tp - t) <= tol_frames:
                candidates.append((abs(tp - t), j))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, jbest = candidates[0]
            pr_used.add(jbest)
            correct += 1

    total_gt = len(gt_tr)
    total_pr = len(pr_tr)
    recall = correct / total_gt if total_gt else 0.0
    precision = correct / total_pr if total_pr else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0

    return {
        "tol_frames": tol_frames,
        "gt_transitions": total_gt,
        "pred_transitions": total_pr,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
