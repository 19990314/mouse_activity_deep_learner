import numpy as np

import numpy as np
from scipy.io import loadmat

def load_speed_w_from_combined(mat_path, session_name=None, session_index=None):
    m = loadmat(mat_path)
    allTrackData = m["allTrackData"]  # shape (1, N)

    if session_name is None and session_index is None:
        # print available session names
        names = [np.squeeze(allTrackData[0, i]["name"]).item() for i in range(allTrackData.shape[1])]
        raise ValueError("Provide session_name or session_index. Available names:\n" + "\n".join(names))

    if session_index is None:
        # find by name
        idx = None
        for i in range(allTrackData.shape[1]):
            nm = np.squeeze(allTrackData[0, i]["name"]).item()
            if nm == session_name:
                idx = i
                break
        if idx is None:
            names = [np.squeeze(allTrackData[0, i]["name"]).item() for i in range(allTrackData.shape[1])]
            raise ValueError(f"session_name not found: {session_name}\nAvailable:\n" + "\n".join(names))
    else:
        idx = int(session_index)

    entry = allTrackData[0, idx]

    speed = np.squeeze(entry["speed_pixels_per_frame"]).astype(float)
    w = np.squeeze(entry["w"]).astype(float)  # angular velocity (likely deg/s scale)

    fps = float(np.squeeze(entry["fps"]))
    nframes = int(np.squeeze(entry["nframes"]))
    name = np.squeeze(entry["name"]).item()

    return {"name": name, "fps": fps, "nframes": nframes, "speed": speed, "w": w}


def _log_gauss(x, mu, sigma):
    sigma = max(1e-6, float(sigma))
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def emission_scores_from_kinematics(speed, angvel, state_ids):
    """
    Build per-frame log-scores E[t,k] using simple kinematics priors.
    Tune parameters to match your units (px/frame vs cm/s, deg/s vs rad/s).
    """
    T = len(speed)
    K = len(state_ids)
    E = np.zeros((T, K), dtype=float)

    abs_w = np.abs(angvel)

    # Robust scale estimates (percentiles) so it adapts across sessions/units
    sp50 = np.nanpercentile(speed, 50)
    sp90 = np.nanpercentile(speed, 90)
    w50  = np.nanpercentile(abs_w, 50)
    w90  = np.nanpercentile(abs_w, 90)

    # Guard against degenerate cases
    sp90 = max(sp90, 1e-6)
    w90  = max(w90, 1e-6)

    for k, s in enumerate(state_ids):
        if s == 2:  # still
            # Prefer near-zero speed (relative to typical movement in this session)
            E[:, k] = _log_gauss(speed, 0.0, 0.15 * sp90)

        elif s == 5:  # groom
            # Low speed, can have some angular movement
            E[:, k] = _log_gauss(speed, 0.1 * sp50, 0.25 * sp90)

        elif s == 4:  # rear
            # Often low locomotion; kinematics alone is weak but still useful
            E[:, k] = _log_gauss(speed, 0.2 * sp50, 0.35 * sp90)

        elif s == 0:  # turn
            # Prefer high |angvel|
            E[:, k] = _log_gauss(abs_w, 0.9 * w90, 0.35 * w90)

        elif s == 1:  # forward
            # Prefer higher speed + lower turning
            E[:, k] = (
                _log_gauss(speed, 0.8 * sp90, 0.35 * sp90)
                + _log_gauss(abs_w, 0.0, 0.45 * w90)
            )

        elif s == 3:  # explore
            # Moderate speed + moderate turning (session-adaptive)
            E[:, k] = (
                _log_gauss(speed, sp50, 0.45 * sp90)
                + _log_gauss(abs_w, w50, 0.55 * w90)
            )
        else:
            # For -1 unsigned or anything else: neutral
            E[:, k] = -1.0

    return E


def viterbi_decode(E, switch_penalty=2.0):
    """
    Viterbi decode for a simple HMM with uniform transition costs:
      stay: 0, switch: -switch_penalty.
    """
    T, K = E.shape
    dp = np.full((T, K), -np.inf, dtype=float)
    back = np.zeros((T, K), dtype=int)

    trans = -switch_penalty * (np.ones((K, K)) - np.eye(K))

    dp[0] = E[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = E[t] + scores[back[t], np.arange(K)]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]

    return path


def refine_states_with_kinematics(pred_states, speed, angvel, switch_penalty=2.0):
    """
    pred_states: array-like of per-frame state IDs (0..5 or -1)
    Returns refined state IDs for the aligned range.
    """
    pred_states = np.asarray(pred_states, dtype=int)

    # Only refine frames where we have kinematics
    T = min(len(pred_states), len(speed), len(angvel))
    pred_states = pred_states[:T]
    speed = speed[:T]
    angvel = angvel[:T]

    # Refine only among the 6 behavior classes; keep -1 as allowed but discouraged
    state_ids = [0, 1, 2, 3, 4, 5, -1]

    E = emission_scores_from_kinematics(speed, angvel, state_ids)

    # Optional: bias toward original prediction to avoid overcorrecting
    # Add a small bonus when the refined label matches original
    bonus = 0.6
    for i, sid in enumerate(state_ids):
        E[np.arange(T), i] += bonus * (pred_states == sid)

    idx_path = viterbi_decode(E, switch_penalty=switch_penalty)
    refined = np.array([state_ids[i] for i in idx_path], dtype=int)

    return refined


kin = load_speed_w_from_combined(r"\\moorelaboratory.dts.usc.edu\Shared\Shuting\P1-SNr\B4_cohort_2_post_injection_bahavior\stats_and_analysis\of\combined_data.mat",
                                 session_name="sc04_d1_of_auto_vTrack_angle.mat")
speed = kin["speed"]
angvel = kin["w"]

print(speed, angvel)
