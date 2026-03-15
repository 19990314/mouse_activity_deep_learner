#!/usr/bin/env python3
"""
verify_of_stats.py — Double-check open-field statistics from DLC .h5 files.

For each h5 file, computes (from frames > 5400 only):
  - MedianSpeed        : median frame-to-frame speed of "nose" (pixels/frame)
  - MeanAngularVelocity: mean |angular velocity| of body axis (nose→tailbase) (rad/frame)
  - n_frames           : number of valid frames used (total frames - 5400)
  - TimeInCenter (sec) : time spent in inner 50% of arena (seconds, at 30 fps)
  - TimeInCenter (%)   : fraction of time in center

Arena bounds are auto-detected per session from the nose trajectory extremes.
The script verifies these form a roughly rectangular arena before computing center zone.

Usage:
  python verify_of_stats.py <path_to_h5_file>

  The script will:
    1. Parse ANIMALID and ExperimentDay from the h5 filename
    2. Look for an existing CSV in the same directory (matching the subject prefix)
    3. Compute stats from the h5, then print comparison with CSV values

  You can also batch-process by passing a directory:
    python verify_of_stats.py <directory_with_h5_files>
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================
FPS = 30
SKIP_FRAMES = 5400  # first 3 minutes are setup, not real data
CONF_THRESHOLD = 0.6  # DLC likelihood threshold for valid tracking
CENTER_FRACTION = 0.5  # inner 50% of arena width/height


# ============================================================
# Filename parsing
# ============================================================
def parse_h5_filename(h5_path: str):
    """
    Parse ANIMALID and ExperimentDay from h5 filename.

    Example: sc09_d4_ofDLC_Resnet101_of_keypointsOct20shuffle5_snapshot_1170_filtered.h5
    → ANIMALID = "sc09", ExperimentDay = 4
    """
    basename = os.path.basename(h5_path)

    # Match pattern: scXX_dN_of...
    m = re.match(r'(sc\d+)_d(\d+)_of', basename)
    if m:
        return m.group(1), int(m.group(2))

    # Fallback: try broader pattern
    m = re.match(r'(sc\d+)_d(\d+)', basename)
    if m:
        return m.group(1), int(m.group(2))

    raise ValueError(f"Could not parse ANIMALID/ExperimentDay from: {basename}")


def get_subject_prefix(h5_path: str) -> str:
    """Get the subject prefix (e.g., 'sc09_d4_of') from h5 filename."""
    basename = os.path.basename(h5_path)
    m = re.match(r'(sc\d+_d\d+_of)', basename)
    if m:
        return m.group(1)
    m = re.match(r'(sc\d+_d\d+)', basename)
    if m:
        return m.group(1)
    return basename.split('DLC')[0].rstrip('_')


# ============================================================
# DLC h5 loading
# ============================================================
def load_dlc_h5(path: str) -> pd.DataFrame:
    """Load DLC h5 and return DataFrame with MultiIndex columns."""
    df = pd.read_hdf(path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DLC .h5 did not load with MultiIndex columns.")
    return df


def extract_bodypart_xy(dlc_df: pd.DataFrame, bodypart: str, conf_thr: float = CONF_THRESHOLD):
    """
    Extract x, y coordinates for a bodypart, interpolating low-confidence frames.

    Returns:
        xy: (T, 2) float32 array
        conf: (T,) float32 array of likelihoods
    """
    # Get scorer level (first level of MultiIndex)
    scorer = dlc_df.columns.get_level_values(0)[0]

    x = dlc_df[(scorer, bodypart, 'x')].to_numpy(dtype=np.float64)
    y = dlc_df[(scorer, bodypart, 'y')].to_numpy(dtype=np.float64)
    conf = dlc_df[(scorer, bodypart, 'likelihood')].to_numpy(dtype=np.float64)

    # Interpolate low-confidence points
    xy = np.column_stack([x, y])
    mask_bad = conf < conf_thr
    if mask_bad.any():
        for j in range(2):
            s = pd.Series(xy[:, j])
            s[mask_bad] = np.nan
            xy[:, j] = s.interpolate(limit_direction='both').to_numpy()

    return xy.astype(np.float64), conf


def find_bodypart(dlc_df: pd.DataFrame, candidates: list) -> str:
    """Find the first available bodypart from a list of candidates."""
    scorer = dlc_df.columns.get_level_values(0)[0]
    available = set(dlc_df.columns.get_level_values(1))
    for bp in candidates:
        if bp in available:
            return bp
    raise ValueError(
        f"None of {candidates} found in h5. Available bodyparts: {sorted(available)}"
    )


# ============================================================
# Statistics computation
# ============================================================
def compute_speed(xy: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame speed from (T, 2) position array.
    Returns (T-1,) array of speeds in pixels/frame.
    """
    dxy = np.diff(xy, axis=0)  # (T-1, 2)
    speed = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2)
    return speed


def compute_body_axis_angular_velocity(nose_xy: np.ndarray, tail_xy: np.ndarray) -> np.ndarray:
    """
    Compute angular velocity from the body axis (nose → tailbase).

    The heading angle at each frame is atan2(nose_y - tail_y, nose_x - tail_x).
    Angular velocity = diff(heading), wrapped to [-pi, pi].

    Returns (T-1,) array of angular velocities in rad/frame.
    """
    dx = nose_xy[:, 0] - tail_xy[:, 0]
    dy = nose_xy[:, 1] - tail_xy[:, 1]
    heading = np.arctan2(dy, dx)  # (T,)

    # Compute angular difference, wrapping to [-pi, pi]
    dtheta = np.diff(heading)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    return dtheta


def detect_arena_bounds(nose_xy: np.ndarray):
    """
    Detect arena rectangle from nose trajectory extremes.

    Finds the four corners where the nose reached farthest in x and y,
    verifies they form a roughly rectangular arena.

    Returns:
        x_min, x_max, y_min, y_max: arena bounds
        is_rectangular: bool — True if corners form a reasonable rectangle
    """
    x = nose_xy[:, 0]
    y = nose_xy[:, 1]

    # Use percentiles (1st and 99th) to be robust to outliers
    x_min = np.percentile(x, 1)
    x_max = np.percentile(x, 99)
    y_min = np.percentile(y, 1)
    y_max = np.percentile(y, 99)

    # Check rectangularity: width and height should be reasonably similar
    # (within a factor of ~2 for a typical open field)
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = max(width, height) / max(min(width, height), 1e-6)

    is_rectangular = aspect_ratio < 2.5  # allow up to 2.5:1 aspect ratio

    return x_min, x_max, y_min, y_max, is_rectangular


def compute_time_in_center(nose_xy: np.ndarray, center_frac: float = CENTER_FRACTION):
    """
    Compute time spent in the center zone of the arena.

    Center zone = inner `center_frac` of both width and height.
    For center_frac=0.5: middle 50% of each dimension (25% border on each side).

    Returns:
        time_in_center_sec: float (seconds at FPS)
        time_in_center_pct: float (fraction 0–1)
        arena_info: dict with arena bounds and center zone bounds
    """
    x_min, x_max, y_min, y_max, is_rect = detect_arena_bounds(nose_xy)

    width = x_max - x_min
    height = y_max - y_min

    # Center zone: inner center_frac of each dimension
    border_frac = (1.0 - center_frac) / 2.0
    cx_min = x_min + border_frac * width
    cx_max = x_max - border_frac * width
    cy_min = y_min + border_frac * height
    cy_max = y_max - border_frac * height

    # Count frames in center
    in_center = (
            (nose_xy[:, 0] >= cx_min) & (nose_xy[:, 0] <= cx_max) &
            (nose_xy[:, 1] >= cy_min) & (nose_xy[:, 1] <= cy_max)
    )

    n_center = int(in_center.sum())
    n_total = len(nose_xy)

    time_sec = n_center / FPS
    time_pct = n_center / max(n_total, 1)

    arena_info = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'width': width, 'height': height,
        'aspect_ratio': max(width, height) / max(min(width, height), 1e-6),
        'is_rectangular': is_rect,
        'center_x_range': (cx_min, cx_max),
        'center_y_range': (cy_min, cy_max),
        'n_center_frames': n_center,
        'n_total_frames': n_total,
    }

    return time_sec, time_pct, arena_info


def compute_all_stats(h5_path: str) -> dict:
    """
    Compute all statistics from a single h5 file.

    All stats use frames after the 5400th frame.
    """
    animal_id, exp_day = parse_h5_filename(h5_path)

    print(f"\n{'=' * 60}")
    print(f"[Processing] {os.path.basename(h5_path)}")
    print(f"  ANIMALID={animal_id}  ExperimentDay={exp_day}")

    # Load DLC data
    dlc_df = load_dlc_h5(h5_path)
    total_frames = len(dlc_df)
    print(f"  Total frames in h5: {total_frames}")

    if total_frames <= SKIP_FRAMES:
        print(f"  [WARN] Only {total_frames} frames, need >{SKIP_FRAMES}. Skipping.")
        return None

    # Trim to experimental frames only
    dlc_df = dlc_df.iloc[SKIP_FRAMES:]
    n_frames = len(dlc_df)
    print(f"  Frames after skipping first {SKIP_FRAMES}: {n_frames}")

    # Find body parts
    nose_name = find_bodypart(dlc_df, ['nose', 'snout', 'Nose', 'Snout'])
    tail_name = find_bodypart(dlc_df, ['tail_base', 'tailbase', 'Tail_base', 'TailBase',
                                       'tail base', 'body_center', 'spine3', 'spine_3'])
    print(f"  Using bodyparts: nose='{nose_name}', tail='{tail_name}'")

    # Extract coordinates
    nose_xy, nose_conf = extract_bodypart_xy(dlc_df, nose_name)
    tail_xy, tail_conf = extract_bodypart_xy(dlc_df, tail_name)

    # --- MedianSpeed ---
    # Speed from nose displacement, frame-to-frame
    speed = compute_speed(tail_xy)
    median_speed = float(np.median(speed))
    print(f"  MedianSpeed (px/frame): {median_speed:.4f}")

    # --- MeanAngularVelocity ---
    # Body axis angular velocity: nose → tailbase heading change
    ang_vel = compute_body_axis_angular_velocity(nose_xy, tail_xy)
    mean_ang_vel = float(np.mean(np.abs(ang_vel)))
    print(f"  MeanAngularVelocity (|rad/frame|): {mean_ang_vel:.6f}")

    # --- TimeInCenter ---
    time_center_sec, time_center_pct, arena_info = compute_time_in_center(nose_xy)

    print(f"  Arena bounds: x=[{arena_info['x_min']:.1f}, {arena_info['x_max']:.1f}]  "
          f"y=[{arena_info['y_min']:.1f}, {arena_info['y_max']:.1f}]")
    print(f"  Arena size: {arena_info['width']:.1f} x {arena_info['height']:.1f} px  "
          f"(aspect ratio: {arena_info['aspect_ratio']:.2f})")
    if not arena_info['is_rectangular']:
        print(f"  [WARN] Arena aspect ratio {arena_info['aspect_ratio']:.2f} > 2.5 — "
              f"may not be a proper rectangle!")
    print(f"  Center zone (inner {CENTER_FRACTION * 100:.0f}%): "
          f"x=[{arena_info['center_x_range'][0]:.1f}, {arena_info['center_x_range'][1]:.1f}]  "
          f"y=[{arena_info['center_y_range'][0]:.1f}, {arena_info['center_y_range'][1]:.1f}]")
    print(f"  TimeInCenter: {time_center_sec:.2f} sec  ({time_center_pct * 100:.1f}%)")
    print(f"  n_frames: {n_frames}")

    return {
        'ANIMALID': animal_id,
        'ExperimentDay': exp_day,
        'MedianSpeed': median_speed,
        'MeanAngularVelocity': mean_ang_vel,
        'n_frames': n_frames,
        'TimeInCenter_sec': time_center_sec,
        'TimeInCenter_pct': time_center_pct,
        # Extra diagnostics
        'arena_width': arena_info['width'],
        'arena_height': arena_info['height'],
        'arena_aspect_ratio': arena_info['aspect_ratio'],
        'is_rectangular': arena_info['is_rectangular'],
    }


# ============================================================
# CSV comparison
# ============================================================
def find_csv_in_directory(h5_path: str) -> str:
    """Find the CSV file in the same directory as the h5 file."""
    h5_dir = os.path.dirname(os.path.abspath(h5_path))
    prefix = get_subject_prefix(h5_path)

    # Try exact prefix match first
    candidates = sorted(glob.glob(os.path.join(h5_dir, '**', '*.csv'), recursive=True))

    # Filter to CSVs that could be the data summary
    for csv_path in candidates:
        csv_name = os.path.basename(csv_path)
        # Skip DLC CSVs (they typically have 'DLC' in the name)
        if 'DLC' in csv_name or 'dlc' in csv_name:
            continue
        # Skip binary files masquerading as .csv (e.g., PNG, Excel)
        try:
            with open(csv_path, 'rb') as f:
                header_bytes = f.read(4)
            # PNG magic: \x89PNG, Excel: \x50\x4b (PK zip)
            if header_bytes[:1] in (b'\x89', b'\x50', b'\xff', b'\x00'):
                continue
        except Exception:
            continue
        # Check if it has the expected columns
        try:
            df = pd.read_csv(csv_path, nrows=2, encoding='utf-8')
            if 'ANIMALID' in df.columns and 'MedianSpeed' in df.columns:
                return csv_path
        except (UnicodeDecodeError, pd.errors.ParserError):
            # Also try latin-1 encoding
            try:
                df = pd.read_csv(csv_path, nrows=2, encoding='latin-1')
                if 'ANIMALID' in df.columns and 'MedianSpeed' in df.columns:
                    return csv_path
            except Exception:
                continue
        except Exception:
            continue

    return None


def compare_with_csv(stats: dict, csv_path: str):
    """
    Compare computed stats with values in the existing CSV.
    """
    # Guard against binary files
    try:
        with open(csv_path, 'rb') as f:
            if f.read(1) in (b'\x89', b'\x50', b'\xff', b'\x00'):
                print(f"\n  [ERROR] {csv_path} appears to be a binary file, not a CSV.")
                return
    except Exception:
        pass

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except Exception as e:
            print(f"\n  [ERROR] Could not read CSV {csv_path}: {e}")
            return

    # Handle duplicate column names (TimeInCenter appears twice)
    # pandas will auto-rename to TimeInCenter, TimeInCenter.1
    cols = list(df.columns)

    # Find matching row
    animal_id = stats['ANIMALID']
    exp_day = stats['ExperimentDay']

    mask = (df['ANIMALID'] == animal_id) & (df['ExperimentDay'] == exp_day)
    if not mask.any():
        print(f"\n  [WARN] No matching row in CSV for {animal_id} day {exp_day}")
        return

    row = df[mask].iloc[0]

    print(f"\n  {'─' * 55}")
    print(f"  COMPARISON with CSV: {os.path.basename(csv_path)}")
    print(f"  {'─' * 55}")
    print(f"  {'Metric':<25s} {'CSV':>12s} {'Computed':>12s} {'Diff%':>8s}")
    print(f"  {'─' * 55}")

    comparisons = [
        ('MedianSpeed', 'MedianSpeed'),
        ('MeanAngularVelocity', 'MeanAngularVelocity'),
        ('n_frames', 'n_frames'),
    ]

    for csv_col, stat_key in comparisons:
        if csv_col in row.index:
            csv_val = float(row[csv_col])
            comp_val = float(stats[stat_key])
            if csv_val != 0:
                diff_pct = (comp_val - csv_val) / abs(csv_val) * 100
            else:
                diff_pct = 0 if comp_val == 0 else float('inf')

            flag = '  ✓' if abs(diff_pct) < 5 else '  ✗' if abs(diff_pct) > 20 else '  ~'
            print(f"  {csv_col:<25s} {csv_val:>12.4f} {comp_val:>12.4f} {diff_pct:>+7.1f}%{flag}")

    # Handle TimeInCenter (may be duplicated)
    tic_cols = [c for c in cols if 'TimeInCenter' in c]
    if len(tic_cols) >= 2:
        # First TimeInCenter is likely seconds, second is likely fraction/percentage
        tic1_val = float(row[tic_cols[0]])
        tic2_val = float(row[tic_cols[1]])

        # Determine which is seconds vs percentage
        # Heuristic: if one is < 1, it's probably a fraction; if > 1, it's seconds
        if tic1_val > 1 and tic2_val <= 1:
            csv_sec, csv_pct = tic1_val, tic2_val
        elif tic2_val > 1 and tic1_val <= 1:
            csv_sec, csv_pct = tic2_val, tic1_val
        else:
            # Both same magnitude — first is sec, second is pct (per your column order)
            csv_sec, csv_pct = tic1_val, tic2_val

        comp_sec = stats['TimeInCenter_sec']
        comp_pct = stats['TimeInCenter_pct']

        if csv_sec != 0:
            diff_sec = (comp_sec - csv_sec) / abs(csv_sec) * 100
        else:
            diff_sec = 0
        if csv_pct != 0:
            diff_pct = (comp_pct - csv_pct) / abs(csv_pct) * 100
        else:
            diff_pct = 0

        flag_s = '  ✓' if abs(diff_sec) < 5 else '  ✗' if abs(diff_sec) > 20 else '  ~'
        flag_p = '  ✓' if abs(diff_pct) < 5 else '  ✗' if abs(diff_pct) > 20 else '  ~'
        print(f"  {'TimeInCenter (sec)':<25s} {csv_sec:>12.4f} {comp_sec:>12.4f} {diff_sec:>+7.1f}%{flag_s}")
        print(f"  {'TimeInCenter (%)':<25s} {csv_pct:>12.4f} {comp_pct:>12.4f} {diff_pct:>+7.1f}%{flag_p}")
    elif len(tic_cols) == 1:
        csv_tic = float(row[tic_cols[0]])
        # Guess if it's seconds or fraction
        if csv_tic > 1:
            comp_val = stats['TimeInCenter_sec']
            label = 'TimeInCenter (sec)'
        else:
            comp_val = stats['TimeInCenter_pct']
            label = 'TimeInCenter (%)'
        if csv_tic != 0:
            diff = (comp_val - csv_tic) / abs(csv_tic) * 100
        else:
            diff = 0
        flag = '  ✓' if abs(diff) < 5 else '  ✗' if abs(diff) > 20 else '  ~'
        print(f"  {label:<25s} {csv_tic:>12.4f} {comp_val:>12.4f} {diff:>+7.1f}%{flag}")

    print(f"  {'─' * 55}")
    print(f"  Legend: ✓ <5% diff  ~ 5-20% diff  ✗ >20% diff")


# ============================================================
# Main
# ============================================================
def process_single_h5(h5_path: str, csv_path: str = None):
    """Process a single h5 file and optionally compare with CSV."""
    stats = compute_all_stats(h5_path)
    if stats is None:
        return None, None

    # Find CSV if not provided
    if csv_path is None:
        csv_path = find_csv_in_directory(h5_path)

    old_row = None
    if csv_path:
        compare_with_csv(stats, csv_path)
        # Also extract the matching row from old CSV for merging
        old_row = _get_old_csv_row(stats, csv_path)
    else:
        print(f"\n  [INFO] No matching CSV found in {os.path.dirname(h5_path)}")

    return stats, old_row


def _get_old_csv_row(stats: dict, csv_path: str) -> dict:
    """Extract the matching row from the old CSV as a dict."""
    try:
        with open(csv_path, 'rb') as f:
            if f.read(1) in (b'\x89', b'\x50', b'\xff', b'\x00'):
                return None
    except Exception:
        return None

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except Exception:
            return None
    except Exception:
        return None

    if 'ANIMALID' not in df.columns or 'ExperimentDay' not in df.columns:
        return None

    mask = (df['ANIMALID'] == stats['ANIMALID']) & (df['ExperimentDay'] == stats['ExperimentDay'])
    if not mask.any():
        return None

    row = df[mask].iloc[0]
    return row.to_dict()


def _merge_old_and_new(old_row: dict, new_stats: dict) -> dict:
    """
    Merge old CSV columns with new computed stats.

    Old CSV columns keep their original names.
    New computed columns get a 'new_' prefix for the overlapping metrics
    so you can compare side by side.
    """
    merged = {}

    # Start with all old CSV columns (if available)
    if old_row:
        for k, v in old_row.items():
            merged[k] = v
    else:
        # No old CSV row — just include ANIMALID and ExperimentDay
        merged['ANIMALID'] = new_stats['ANIMALID']
        merged['ExperimentDay'] = new_stats['ExperimentDay']

    # Add new computed values with 'new_' prefix for the columns that overlap
    overlap_keys = {'MedianSpeed', 'MeanAngularVelocity', 'n_frames'}
    for k, v in new_stats.items():
        if k in ('ANIMALID', 'ExperimentDay'):
            continue  # already in merged from old row
        if k in overlap_keys:
            merged[f'new_{k}'] = v
        elif k == 'TimeInCenter_sec':
            merged['new_TimeInCenter_sec'] = v
        elif k == 'TimeInCenter_pct':
            merged['new_TimeInCenter_pct'] = v
        else:
            # Arena diagnostics etc. — no prefix needed
            merged[k] = v

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Double-check open-field stats from DLC h5 files against existing CSV."
    )
    parser.add_argument(
        'path',
        help="Path to a single .h5 file, or a directory containing .h5 files"
    )
    parser.add_argument(
        '--csv', default=None,
        help="Path to the CSV with existing stats (auto-detected if not provided)"
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help="Frame rate (default: 30)"
    )
    parser.add_argument(
        '--skip', type=int, default=5400,
        help="Number of initial frames to skip (default: 5400 = 3 min at 30fps)"
    )
    parser.add_argument(
        '--center_frac', type=float, default=0.5,
        help="Center zone fraction (default: 0.5 = inner 50%%)"
    )
    parser.add_argument(
        '--conf_thr', type=float, default=0.6,
        help="DLC confidence threshold for valid tracking (default: 0.6)"
    )

    args = parser.parse_args()

    # Update globals
    global FPS, SKIP_FRAMES, CENTER_FRACTION, CONF_THRESHOLD
    FPS = args.fps
    SKIP_FRAMES = args.skip
    CENTER_FRACTION = args.center_frac
    CONF_THRESHOLD = args.conf_thr

    target = args.path

    if os.path.isfile(target) and target.endswith('.h5'):
        # Single file
        stats, old_row = process_single_h5(target, args.csv)
        if stats:
            merged = _merge_old_and_new(old_row, stats)
            merged["new_MedianSpeed_cm_p_s"] = (merged["new_MedianSpeed"]*30)/merged["pixels_per_cm"]
            print("???")
            h5_dir = os.path.dirname(os.path.abspath(target))
            out_csv = os.path.join(h5_dir, 'verify_of_stats_results.csv')
            pd.DataFrame([merged]).to_csv(out_csv, index=False)
            print(f"\n[SAVED] {out_csv}")
    elif os.path.isdir(target):
        # Directory: process all h5 files
        h5_files = sorted(glob.glob(os.path.join(target, '**', '*DLC*.h5'), recursive=True))
        if not h5_files:
            h5_files = sorted(glob.glob(os.path.join(target, '**', '*.h5'), recursive=True))

        if not h5_files:
            print(f"No .h5 files found in {target}")
            sys.exit(1)

        print(f"Found {len(h5_files)} h5 file(s) in {target}")

        all_merged = []
        for h5_path in h5_files:
            stats, old_row = process_single_h5(h5_path, args.csv)
            if stats:
                all_merged.append(_merge_old_and_new(old_row, stats))

        # Print summary table
        if all_merged:
            print(f"\n\n{'=' * 80}")
            print("SUMMARY")
            print(f"{'=' * 80}")
            print(f"{'ANIMALID':<10s} {'Day':>4s} {'MedianSpd':>10s} {'MeanAngVel':>11s} "
                  f"{'n_frames':>9s} {'TIC_sec':>8s} {'TIC_%':>7s} {'Arena':>12s}")
            print(f"{'─' * 80}")
            for s in all_merged:
                rect_flag = '✓' if s.get('is_rectangular') else '✗'
                print(f"{s['ANIMALID']:<10s} {s['ExperimentDay']:>4d} "
                      f"{s['new_MedianSpeed']:>10.4f} {s['new_MeanAngularVelocity']:>11.6f} "
                      f"{s['new_n_frames']:>9d} {s['new_TimeInCenter_sec']:>8.2f} "
                      f"{s['new_TimeInCenter_pct'] * 100:>6.1f}% "
                      f"{s.get('arena_width', 0):.0f}x{s.get('arena_height', 0):.0f}{rect_flag}")

            # Save results to CSV
            out_csv = os.path.join(target, 'verify_of_stats_results.csv')
            df_out = pd.DataFrame(all_merged)
            df_out["new_MedianSpeed_cm_p_s"] = (df_out["new_MedianSpeed"] * 30) / df_out["pixels_per_cm"]
            df_out.to_csv(out_csv, index=False)
            print(f"\n[SAVED] {out_csv}  ({len(all_merged)} rows)")
    else:
        print(f"Error: {target} is not a valid .h5 file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()