#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate per-frame state numbers onto a video.

Usage example:
    python stamp_states_on_video.py \
        --video m3-d7-openfield.mp4 \
        --states states_per_frame.csv \
        --column state \
        --out m3-d7-openfield_annotated.mp4
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from scipy.io import loadmat
from evaluate_annotation import *

merged_labels = {
    0: "turn",
    1: "forward",
    2: "still",
    3: "explore",
    4: "rear",
    5: "groom",
    -1: "unsigned"
}
low_confident_syllables = [18, 26, 6, 3, 10, 4, 16]

label_durations = {"still": 40, "turn": 10, "rear": 15, "groom": 20, "forward": 30, "explore": 15}

def mapping_and_merge(df):
    syllables_to_merge = [
        #[10, 8, 16, 20, 22, 27, 44, 45, 66, 89],  # turn
        #[0, 30, 4, 34, 2, 9, 17, 12, 19, 99, 90, 31],  # still
        #[40, 95, 97, 20, 1, 3, 6, 24, 58],  # forward
        #[32, 57, 61, 19, 69],  # explore
        #[5, 33, 35, 67, 81, 58],  # rear
        #[68, 31]  # groom
        [3,7,8,9,23], # turn
        [5,6,11,12,14, 15, 16],  # forward: 4
        [2, 27],  # still
        [0,4,13,17,20, 21],  # explore: 20, 6
        [10,18, 19,22, 24],  # rear: 17, 19, 21
        [16,25, 26]  # groom
    ]
    # 18, 26, 6, 3, 10, 4

    # Optional: label names for readability

    # --------------------------
    # Build syllable → merged_id map
    # --------------------------
    mapping = {}
    for merged_id, group in enumerate(syllables_to_merge):
        for s in group:
            mapping[s] = merged_id

    original = df['syllable'].to_numpy()
    print(np.unique(original))

    # --------------------------
    # Apply mapping
    # --------------------------
    merged = []
    for s in original:
        if s in mapping:
            merged.append(mapping[s])
        else:
            merged.append(-1)

    df["syllable_merged"] = merged
    df = smooth_short_syllable_blocks(df, col="syllable_merged", min_len=10, n_neighbors=12)

    for label in label_durations.keys():
        first_key = next((k for k, v in merged_labels.items() if v == label), None)
        df = smooth_short_label_blocks(
            df,
            col="syllable_merged",
            target_label=first_key,
            min_len=label_durations[label],
            n_neighbors=12
        )

    df["syllable_merged_label"] = df["syllable_merged"].map(merged_labels)
    return df


def smooth_short_label_blocks(
    df,
    col="syllable_merged",
    target_label=0,
    min_len=30,
    n_neighbors=40
):
    """
    For a label column (e.g. 'syllable_merged_label'):
      - find contiguous blocks where label == target_label (e.g. 'still')
      - if block length < min_len, replace it with the most frequent
        *neighboring label* in a window of n_neighbors around the block.

    Neighbors are taken from both sides of the block (excluding the block itself).
    Preference is given to non-target labels, to avoid just re-assigning 'still' to 'still'.
    """
    labels = df[col].to_numpy().copy()
    n = len(labels)
    half = n_neighbors // 2
    is_confident = ~df["syllable"].isin(low_confident_syllables)

    start = 0
    while start < n:
        val = labels[start]
        end = start + 1
        # find contiguous block of the same label
        while end < n and labels[end] == val:
            end += 1

        block_len = end - start

        # Only operate on the target label (e.g. 'still')
        if val == target_label and block_len < min_len:
            print("found: "+merged_labels[target_label]+" < " +str(block_len))
            left = max(0, start - half)
            right = min(n, end + half)

            neighbors = np.concatenate([
                labels[left:start][is_confident.iloc[left:start].values],
                labels[end:right][is_confident.iloc[end:right].values]
            ])

            # drop NaNs or empties
            #neighbors = np.array([x for x in neighbors if isinstance(x, str) or x == x])

            if neighbors.size > 0:
                counts = Counter(neighbors.tolist())

                repl, _ = counts.most_common(1)[0]
                labels[start:end] = repl

        start = end

    df[col] = labels
    return df


def smooth_short_syllable_blocks(df, col="syllable", min_len=10, n_neighbors=20):
    """
    For a given 'syllable' column in df:
      - find contiguous runs (blocks) of the same value
      - if a block length < min_len, replace it with the most frequent
        syllable in a window of n_neighbors around the block (excluding the block itself).

    Args:
        df: pandas DataFrame containing the syllable column.
        col: name of the syllable column to process.
        min_len: minimum length of a block to be kept as-is.
        n_neighbors: total number of neighbor frames to use around the block
                     (we use ~n_neighbors/2 before and after).
    Returns:
        df with the column modified in-place and also returned for convenience.
    """
    syll = df[col].to_numpy().copy()
    n = len(syll)
    half = n_neighbors // 2

    start = 0
    while start < n:
        val = syll[start]
        end = start + 1
        # find end of this contiguous block
        while end < n and syll[end] == val:
            end += 1

        block_len = end - start

        if block_len < min_len:
            # define neighbor window around the block
            left = max(0, start - half)
            right = min(n, end + half)

            # neighbors are everything in [left:start) and (end:right]
            neighbors = np.concatenate([
                syll[left:start],
                syll[end:right]
            ])

            # if we have neighbors, pick the most frequent label
            if neighbors.size > 0:
                counts = Counter(neighbors.tolist())
                # choose the most common neighbor syllable
                repl, _ = counts.most_common(1)[0]
                syll[start:end] = repl

        start = end

    df[col] = syll
    return df


def annotate_video(
    video_path: str,
    states_csv: str,
    column: str,
    out_path: str = None,
    font_scale: float = 0.8,
    thickness: int = 2,
    margin: int = 15,
):
    # --------- Load states table ----------
    if not os.path.exists(states_csv):
        raise FileNotFoundError(f"State CSV not found: {states_csv}")

    df = pd.read_csv(states_csv)

    # calculate most prob state
    #state_cols = [col for col in df.columns if 'state' in col.lower()]
    #df['state_index'] = df[state_cols].idxmax(axis=1).apply(
    #    lambda x: state_cols.index(x) if x in state_cols else np.nan)
    #states = df["syllable_merged_label"]
    if column == "syllable_merged_label":
        df = mapping_and_merge(df)

    # --- Optional kinematics-based refinement ---
    # Refine a numeric per-frame column (e.g. 'syllable_merged' or your model's 'state' column)
    if args.kinematics_mat is not None and args.refine_column is not None:
        speed, angvel = load_kinematics_mat(args.kinematics_mat, args.speed_key, args.angvel_key)
        refined = refine_states_with_kinematics(df[args.refine_column].to_numpy(), speed, angvel,
                                                switch_penalty=args.switch_penalty)
        df[args.refine_column + "_refined"] = refined
        # Also provide readable labels
        df[args.refine_column + "_refined_label"] = df[args.refine_column + "_refined"].map(merged_labels)

    root, ext = os.path.splitext(video_path)
    df.to_csv(root + "_annotations.csv", index=False)

    no_video_writing_to_save_time_debug = 1
    if no_video_writing_to_save_time_debug == 0:
        states = df[column]
        n_states = len(states)
        print(f"[INFO] Loaded {n_states} frame-wise states from {states_csv}")

        # --------- Open video ----------
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {video_path}")
        print(f"       resolution: {width} x {height}, fps: {fps:.2f}, frames: {n_frames}")

        if out_path is None:
            root, ext = os.path.splitext(video_path)
            #out_path = root + "_annotated.mp4
            out_path = root + "_annotated.avi"

        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        # --------- Annotate frame by frame ----------
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < n_states:
                state_val = states[frame_idx]
                matches = (df["syllable_merged"][0:frame_idx] == df["human_labeled_state"][0:frame_idx]).sum()
                percent_alignment = (matches / frame_idx) * 100

                text = (f"state: {state_val}"
                        + str(df["syllable_merged"][frame_idx])
                        + " vs " + str(merged_labels[df["human_labeled_state"][frame_idx]])
                        + f" Acc: {percent_alignment:.2f}%")

            else:
                # No more states; optionally break instead of leaving blank
                text = "state: NA"

            # Put a filled rectangle behind the text for readability
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x0 = margin
            y0 = margin
            x1 = x0 + text_w + 10
            y1 = y0 + text_h + 10

            # Background box (black, slightly transparent if you want to get fancy)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)

            # White text on top
            cv2.putText(
                frame,
                text,
                (x0 + 5, y0 + text_h + 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 500 == 0:
                print(f"[INFO] Processed {frame_idx}/{n_frames} frames...")

        cap.release()
        writer.release()
        print(f"[DONE] Wrote annotated video to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate frame-wise states onto a video."
    )
    parser.add_argument("--video", required=True, help="Input video (.mp4)")
    parser.add_argument(
        "--states",
        required=True,
        help="CSV file with per-frame state predictions (one row per frame).",
    )
    parser.add_argument(
        "--column",
        default="state",
        help="Column name in CSV containing the state index (default: 'state').",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output video path (.mp4). Default: <input>_annotated.mp4",
    )
    parser.add_argument("--font-scale", type=float, default=0.8)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--margin", type=int, default=15)
    parser.add_argument("--kinematics-mat", default=None, help="Path to .mat with speed & angVel.")
    parser.add_argument("--speed-key", default="speed", help="Variable name for speed in .mat")
    parser.add_argument("--angvel-key", default="angVel", help="Variable name for angular velocity in .mat")
    parser.add_argument("--refine-column", default=None, help="Column to refine (e.g., 'syllable_merged').")
    parser.add_argument("--switch-penalty", type=float, default=2.0, help="Higher = smoother refined sequence.")

    args = parser.parse_args()

    annotate_video(
        video_path=args.video,
        states_csv=args.states,
        column=args.column,
        out_path=args.out,
        font_scale=args.font_scale,
        thickness=args.thickness,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
    calculate_column_alignment(r"D:\My Drive\moseq_proj\data\videos\test\sc04_d3_10mintest_annotations.csv",
                               'human_labeled_state', 'syllable_merged')