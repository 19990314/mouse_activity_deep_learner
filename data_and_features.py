import numpy as np
import pandas as pd
from scipy.io import loadmat
import numpy as np

MIN_LEN = {
    0: 8,   # turn
    1: 15,   # forward
    2: 40,   # still
    3: 6,   # explore
    4: 8,   # rear
    5: 10,   # groom
}


def viterbi_switch_penalty(logp, switch_penalty=2.5):
    """
    logp: (T, K) log-probabilities for K classes.
    Transition: stay cost 0, switch cost -switch_penalty.
    """
    T, K = logp.shape
    dp = np.full((T, K), -np.inf, dtype=np.float64)
    back = np.zeros((T, K), dtype=np.int64)
    trans = -switch_penalty * (np.ones((K, K)) - np.eye(K))

    dp[0] = logp[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = logp[t] + scores[back[t], np.arange(K)]

    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path

def enforce_min_duration(states, min_len_by_class):
    """
    Simple post-process: if a bout shorter than min_len, merge into neighbor with higher support.
    """
    s = states.copy()
    T = len(s)
    i = 0
    while i < T:
        j = i + 1
        while j < T and s[j] == s[i]:
            j += 1
        cls = int(s[i])
        bout_len = j - i
        min_len = int(min_len_by_class.get(cls, 1))
        if bout_len < min_len:
            left = s[i - 1] if i > 0 else None
            right = s[j] if j < T else None
            if left is None and right is None:
                pass
            elif left is None:
                s[i:j] = right
            elif right is None:
                s[i:j] = left
            else:
                s[i:j] = left  # simplest; you can choose based on local probs
        i = j
    return s


STATE_MAP = {0:"turn", 1:"forward", 2:"still", 3:"explore", 4:"rear", 5:"groom", -1:"unsigned"}
VALID_CLASSES = [0, 1, 2, 3, 4, 5]

def load_dlc_csv(dlc_csv_path: str) -> pd.DataFrame:
    """
    Loads DLC CSV with 3-row header. Returns a DataFrame with MultiIndex columns:
    (scorer, bodypart, coord). First column is usually ('scorer','bodyparts','coords') and can be dropped.
    """
    df = pd.read_csv(dlc_csv_path, header=[0, 1, 2])
    # Drop the first column if it's that DLC meta column
    if df.columns[0][0] == "scorer" and df.columns[0][1] == "bodyparts":
        df = df.drop(columns=df.columns[0])
    return df

def load_annotations_csv(ann_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(ann_csv_path)

def load_kinematics_from_combined_mat(mat_path: str, session_name: str):
    """
    combined_data.mat contains allTrackData: (1, N) struct array.
    Each entry includes speed_pixels_per_frame and w.
    """
    m = loadmat(mat_path)
    allTrackData = m["allTrackData"]  # shape (1, N)

    idx = None
    for i in range(allTrackData.shape[1]):
        nm = np.squeeze(allTrackData[0, i]["name"]).item()
        if nm == session_name:
            idx = i
            break
    if idx is None:
        names = [np.squeeze(allTrackData[0, i]["name"]).item() for i in range(allTrackData.shape[1])]
        raise ValueError(f"session_name not found: {session_name}\nAvailable:\n" + "\n".join(names))

    entry = allTrackData[0, idx]
    speed = np.squeeze(entry["speed_pixels_per_frame"]).astype(np.float32)
    w = np.squeeze(entry["w"]).astype(np.float32)  # angular velocity
    fps = float(np.squeeze(entry["fps"]))
    nframes = int(np.squeeze(entry["nframes"]))
    return speed, w, fps, nframes

def interpolate_low_conf(xy: np.ndarray, conf: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """
    xy: (T, 2) array, conf: (T,) array. Low conf -> NaN -> linear interpolation.
    """
    T = xy.shape[0]
    out = xy.copy().astype(np.float32)
    mask = conf < thr
    out[mask] = np.nan

    for j in range(2):
        s = pd.Series(out[:, j])
        out[:, j] = s.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    return out

def compute_velocity(xy: np.ndarray) -> np.ndarray:
    """
    xy: (T, 2). Returns v: (T, 2) with v[0]=0.
    """
    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = xy[1:] - xy[:-1]
    return v

def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    return (x - mu) / (sd + eps)

def window_majority_labels(y: np.ndarray, win: int = 7, ignore_label: int = -1) -> np.ndarray:
    """
    Majority vote in a centered window. If too many ignore labels, keep ignore_label.
    """
    assert win % 2 == 1
    r = win // 2
    T = len(y)
    y_out = y.copy()
    for t in range(T):
        a = max(0, t - r)
        b = min(T, t + r + 1)
        seg = y[a:b]
        seg = seg[seg != ignore_label]
        if len(seg) == 0:
            y_out[t] = ignore_label
        else:
            vals, counts = np.unique(seg, return_counts=True)
            y_out[t] = vals[np.argmax(counts)]
    return y_out

def build_feature_matrix(
    dlc_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    label_col: str = "human_labeled_state",
    dlc_conf_thr: float = 0.6,
    smooth_label_win: int = 7,
    add_moseq: bool = True,
    add_latents: bool = True,
    add_centroid_heading: bool = True,
    speed: np.ndarray | None = None,
    angvel: np.ndarray | None = None,
):
    """
    Returns:
      X: (T, F) float32
      y: (T,) int64
      mask: (T,) bool (True where y is valid)
      feature_names: list[str]
    """
    # Align lengths
    T = min(len(dlc_df), len(ann_df))
    if speed is not None and angvel is not None:
        T = min(T, len(speed), len(angvel))

    dlc_df = dlc_df.iloc[:T]
    ann_df = ann_df.iloc[:T]

    # DLC bodyparts
    bodyparts = sorted(set([c[1] for c in dlc_df.columns]))
    # Remove weird header tokens if any
    bodyparts = [bp for bp in bodyparts if bp not in ("bodyparts", "coords")]

    feats = []
    feat_names = []

    # Pose + velocities + confidence summaries
    conf_all = []
    for bp in bodyparts:
        x = dlc_df.xs((bp, "x"), level=(1, 2), axis=1).to_numpy().squeeze()
        y = dlc_df.xs((bp, "y"), level=(1, 2), axis=1).to_numpy().squeeze()
        p = dlc_df.xs((bp, "likelihood"), level=(1, 2), axis=1).to_numpy().squeeze()

        xy = np.stack([x, y], axis=1).astype(np.float32)
        p = p.astype(np.float32)

        xy = interpolate_low_conf(xy, p, thr=dlc_conf_thr)
        v = compute_velocity(xy)

        feats.append(xy); feat_names += [f"{bp}_x", f"{bp}_y"]
        feats.append(v);  feat_names += [f"{bp}_vx", f"{bp}_vy"]

        conf_all.append(p.reshape(-1, 1))
        feat_names += [f"{bp}_p"]

    conf_mat = np.concatenate(conf_all, axis=1)  # (T, B)
    p_mean = np.mean(conf_mat, axis=1, keepdims=True)
    p_min  = np.min(conf_mat, axis=1, keepdims=True)
    feats.append(p_mean); feat_names += ["p_mean"]
    feats.append(p_min);  feat_names += ["p_min"]

    # MoSeq features
    if add_moseq:
        syll = ann_df["syllable"].to_numpy(dtype=np.int64).reshape(-1, 1)
        feats.append(syll.astype(np.float32)); feat_names += ["syllable_id"]

    if add_latents:
        for j in range(4):
            col = f"latent_state {j}"
            feats.append(ann_df[col].to_numpy(dtype=np.float32).reshape(-1, 1))
            feat_names += [col]

    if add_centroid_heading:
        for col in ["centroid x", "centroid y", "heading"]:
            feats.append(ann_df[col].to_numpy(dtype=np.float32).reshape(-1, 1))
            feat_names += [col]

    # Kinematics from MAT (optional)
    if speed is not None and angvel is not None:
        feats.append(speed[:T].reshape(-1, 1).astype(np.float32));  feat_names += ["speed_pixels_per_frame"]
        feats.append(angvel[:T].reshape(-1, 1).astype(np.float32)); feat_names += ["w_angvel"]

    X = np.concatenate(feats, axis=1).astype(np.float32)
    X = zscore(X)  # session-wise z-score

    y = ann_df[label_col].to_numpy(dtype=np.int64)
    y = y[:T]

    # Smooth labels to the resolution humans can reliably annotate
    if smooth_label_win is not None and smooth_label_win >= 3:
        y = window_majority_labels(y, win=smooth_label_win, ignore_label=-1)

    mask = np.isin(y, VALID_CLASSES)

    return X, y, mask, feat_names
