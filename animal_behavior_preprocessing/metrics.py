"""Computes mouse size, learning and performance metrics."""
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, medfilt
from scipy.stats import trim_mean, norm
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from utilities import read_pickle, write_pickle
import config
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=2)
sns.set_style("white")


def get_latency_to_fall(name, dep_pickle_path, target_pickle_paths):
    """Estimates falling time from volatility of center of mass markers.

    Parameters
    ----------
    name : str
        Mouse, day, trial.
    dep_pickle_path : pathlib.Path
        Path of dependency pickle file to read
    target_pickle_paths : dict of pathlib.Path
        Paths of target pickle files to save
    """
    ys = read_pickle(dep_pickle_path)[:, config.CM_MARKER_IDX, 1]
    volatility = pd.DataFrame(ys).rolling(500).std().mean(axis=1) * 10
    volatility = savgol_filter(volatility, 501, 3, deriv=1) * 500
    volatility = volatility / np.nanmax(volatility) * 100.0
    peaks, properties = find_peaks(volatility, height=15.0, prominence=20.0)
    latency = peaks[-1]
    write_pickle(latency, target_pickle_paths["latency"])

    plt.figure(figsize=(15, 7.5))
    plt.plot(ys[:, 0], alpha=0.9)
    plt.plot(volatility, alpha=0.9)
    plt.plot(peaks, volatility[peaks], "x", alpha=0.9)
    plt.vlines(
        x=peaks,
        ymin=volatility[peaks] - properties["prominences"],
        ymax=volatility[peaks],
        alpha=0.9,
        colors="k",
    )
    plt.title(name)
    plt.savefig(target_pickle_paths["fig_latency"], bbox_inches="tight")
    plt.close()


def get_initial_pairs_t_d2y(y, peaks, valleys):
    """Looks for pairs of peaks and valleys."""
    pairs = []
    for peak in peaks:
        for valley in valleys[valleys > peak]:
            if valley - peak < 50:
                if y[peak] <= y[valley]:
                    pairs.append((peak, valley))
                    break
    return np.array(pairs)


def get_no_repetitions_pairs_t_d2y(pairs):
    """Removes repeating pairs of peaks and valleys."""
    pairs_no_repetitions = []
    for valley in np.unique(pairs[:, 1]):
        repeated_valleys = pairs[:, 1] == valley
        if len(repeated_valleys) == 1:
            pairs_no_repetitions.append(pairs[repeated_valleys])
        else:
            keep = pairs[repeated_valleys][
                np.argmin(np.diff(pairs[repeated_valleys], axis=1))
            ]
            pairs_no_repetitions.append(keep)
    return np.array(pairs_no_repetitions)


def get_no_stacked_pairs_t_d2y(y, pairs_no_repetitions):
    """Removes pairs of peaks and valleys that are stacked."""
    pairs_no_stacked = []
    i = 0
    while i < len(pairs_no_repetitions) - 2:
        pair, next_pair = pairs_no_repetitions[i], pairs_no_repetitions[i + 1]
        if y[next_pair[0]] >= y[pair[1]] and next_pair[0] - pair[1] < 50:
            pairs_no_stacked.append((pair[0], next_pair[1]))
            i += 2
        else:
            pairs_no_stacked.append(pair)
            i += 1
            if i == len(pairs_no_repetitions) - 2:
                pairs_no_stacked.append(next_pair)
    return np.array(pairs_no_stacked)


def get_sliding_pairs_t_y(y, pairs_no_stacked):
    """Returns pairs using the location of the closest extrema."""
    y_pair_window = 7
    y_pairs_with_sliding = []
    for i in range(len(pairs_no_stacked)):
        peak, valley = pairs_no_stacked[i]
        y_pairs_with_sliding.append(
            (
                np.maximum(
                    0,
                    y[
                        np.maximum(0, peak - y_pair_window) : peak + y_pair_window
                    ].argmin()
                    + peak
                    - y_pair_window,
                ),
                np.maximum(
                    0,
                    y[
                        np.maximum(0, valley - y_pair_window) : valley + y_pair_window
                    ].argmax()
                    + valley
                    - y_pair_window,
                ),
            )
        )
    return np.array(y_pairs_with_sliding)


def get_no_interspersed_pairs(pairs_no_stacked, y_pairs_with_sliding):
    """Removes interspersed pairs of peaks and valleys."""
    pairs_to_keep = []
    y_pairs = []
    i = 0
    while i < len(pairs_no_stacked) - 2:
        pair, next_pair = pairs_no_stacked[i], pairs_no_stacked[i + 1]
        y_pair, y_next_pair = y_pairs_with_sliding[i], y_pairs_with_sliding[i + 1]
        if y_pair[1] >= y_next_pair[0]:
            pairs_to_keep.append((pair[0], next_pair[1]))
            y_pairs.append((y_pair[0], y_next_pair[1]))
            i += 2
        else:
            pairs_to_keep.append(pair)
            y_pairs.append(y_pair)
            i += 1
            if i == len(pairs_no_stacked) - 2:
                pairs_to_keep.append(next_pair)
                y_pairs.append(y_next_pair)
    new_pairs = []
    new_y_pairs = []
    for i in range(len(pairs_to_keep)):
        pair = pairs_to_keep[i]
        y_pair = y_pairs[i]
        if y_pair[0] < y_pair[1]:
            new_pairs.append(pair)
            new_y_pairs.append(y_pair)
    return np.array(new_pairs), np.array(new_y_pairs)


def get_t_dy_max(dy, y_pairs):
    """Computes time and values of maximum of dy."""
    t_dy_max = []
    dy_max = []
    for i in range(len(y_pairs)):
        pair = y_pairs[i]
        t_dy_max.append(np.argmax(dy[pair[0] : pair[1]]) + pair[0])
        dy_max.append(dy[t_dy_max[-1]])
    return np.array(t_dy_max), np.array(dy_max)


def get_single_hindpaw_step_events(y):
    """Computes single hindpaw step metrics."""
    dy = savgol_filter(
        y,
        window_length=7,
        polyorder=2,
        deriv=1,
        delta=0.01,
    )
    d2y = savgol_filter(
        y,
        window_length=7,
        polyorder=2,
        deriv=2,
        delta=0.01,
    )
    d2y_noise = np.std(d2y[np.abs(d2y * 0.005) < 5] * 0.005)
    t_d2y1, _ = find_peaks(d2y * 0.005, height=d2y_noise, prominence=d2y_noise, wlen=31)
    t_d2y2, _ = find_peaks(
        -1.0 * d2y * 0.005, height=d2y_noise, prominence=d2y_noise, wlen=31
    )
    pairs_t_d2y = get_initial_pairs_t_d2y(y, t_d2y1, t_d2y2)
    pairs_t_d2y = get_no_repetitions_pairs_t_d2y(pairs_t_d2y)
    pairs_t_d2y = get_no_stacked_pairs_t_d2y(y, pairs_t_d2y)
    pairs_t_y = get_sliding_pairs_t_y(y, pairs_t_d2y)
    pairs_t_d2y, pairs_t_y = get_no_interspersed_pairs(pairs_t_d2y, pairs_t_y)
    t_dy_max, dy_max = get_t_dy_max(dy, pairs_t_y)
    isi_dy_max = np.diff(t_dy_max, prepend=2 * t_dy_max[0] - t_dy_max[1])
    events = {
        "num_steps": len(pairs_t_y),
        "t_y1": pairs_t_y[:, 0],
        "t_y2": pairs_t_y[:, 1],
        "y1": y[pairs_t_y[:, 0]],
        "y2": y[pairs_t_y[:, 1]],
        "t_d2y1": pairs_t_d2y[:, 0],
        "t_d2y2": pairs_t_d2y[:, 1],
        "d2y1": d2y[pairs_t_d2y[:, 0]],
        "d2y2": d2y[pairs_t_d2y[:, 1]],
        "t_dy_max": t_dy_max,
        "dy_max": dy_max,
        "isi_dy_max": isi_dy_max,
    }
    return events


def get_delta_phis(step_events):
    """Computes delta phi for each step, with step_events a pd.DataFrame."""
    delta_phis = []
    hindpaws = list(config.STEP_MARKER_DICT.values())
    for sign_ipsi, paw_ipsi, paw_contra in [
        (1.0, hindpaws[0], hindpaws[1]),
        (-1.0, hindpaws[1], hindpaws[0]),
    ]:
        ipsis = step_events[step_events["hindpaw"] == paw_ipsi]
        contras = step_events[step_events["hindpaw"] == paw_contra]
        for t_ipsi in ipsis["t_dy_max"]:
            t_contra_idx = np.argmin(np.abs(contras["t_dy_max"] - t_ipsi))
            t_contra = contras["t_dy_max"].iloc[t_contra_idx]
            valid_idx = np.where(contras["t_dy_max"] >= t_ipsi)[0]
            if valid_idx.size:
                isi_contra_idx = valid_idx[contras["t_dy_max"].iloc[valid_idx].argmin()]
            else:
                isi_contra_idx = t_contra_idx
            isi_contra = contras["isi_dy_max"].iloc[isi_contra_idx]
            delta_phi = (t_contra - t_ipsi) / isi_contra * sign_ipsi
            delta_phis.append(delta_phi)
    quo, abs_delta_phis = np.divmod(np.abs(delta_phis), 1.0)
    abs_delta_phis[quo >= 1.0] = quo[quo >= 1.0] - abs_delta_phis[quo >= 1.0]
    return delta_phis, abs_delta_phis


def get_moving_window_interpolation(
    t, x, y, window_size=11, interpolation="linear", cut=0.3
):
    """Interpolate a function using a moving window."""
    x_window = np.column_stack(
        [x[i : i + window_size] for i in range(len(x) - window_size + 1)]
    ).T
    mu = trim_mean(x_window, proportiontocut=cut, axis=1)
    sigma = trim_mean(
        np.abs(x_window - mu[:, np.newaxis]), proportiontocut=cut, axis=1
    ) / norm.ppf(0.75)
    t_interpolated = np.arange(len(y))
    t_ini = t[window_size // 2]
    t_fin = t[-window_size // 2 + 1]
    mu_interpolated = interp1d(
        t[window_size // 2 : -window_size // 2 + 1],
        mu,
        fill_value="extrapolate",
        kind="nearest",
    )(t_interpolated)
    mu_interpolated[t_ini:t_fin] = interp1d(
        t[window_size // 2 : -window_size // 2 + 1],
        mu,
        fill_value="extrapolate",
        kind=interpolation,
    )(t_interpolated[t_ini:t_fin])
    sigma_interpolated = interp1d(
        t[window_size // 2 : -window_size // 2 + 1],
        sigma,
        fill_value="extrapolate",
        kind="nearest",
    )(t_interpolated)
    sigma_interpolated[t_ini:t_fin] = interp1d(
        t[window_size // 2 : -window_size // 2 + 1],
        sigma,
        fill_value="extrapolate",
        kind=interpolation,
    )(t_interpolated[t_ini:t_fin])
    return list(mu_interpolated), list(sigma_interpolated)


def get_step_statistics(step_events, y):
    """Computes step statistics."""
    key_dicts = [
        {"t": "t_y1", "x": "y1"},
        {"t": "t_y2", "x": "y2"},
        {"t": "t_d2y1", "x": "d2y1"},
        {"t": "t_d2y2", "x": "d2y2"},
        {"t": "t_dy_max", "x": "dy_max"},
        {"t": "t_dy_max", "x": "isi_dy_max"},
        {"t": "t_dy_max", "x": "delta_phi"},
        {"t": "t_dy_max", "x": "abs_delta_phi"},
        {"t": "t_dy_max", "x": "amp"},
        {"t": "t_dy_max", "x": "freq"},
    ]
    statistics = defaultdict(list)
    for hindpaw in config.STEP_MARKER_DICT.values():
        events = step_events.query("hindpaw == @hindpaw")
        statistics["hindpaw"] += [hindpaw] * len(y)
        for key_dict in key_dicts:
            t = events[key_dict["t"]].to_numpy()
            x = events[key_dict["x"]].to_numpy()
            mu, sigma = get_moving_window_interpolation(t=t, x=x, y=y)
            statistics["mu_" + key_dict["x"]] += mu
            statistics["sigma_" + key_dict["x"]] += sigma
    return statistics


def get_hindpaw_speed_baseline(xys):
    """Computes hindpaw speed baseline."""
    median_baseline_window = 1001
    min_baseline_window = 11
    dy_baseline = []
    for y in [xys[:, idx, 1] for idx in config.STEP_MARKER_DICT.keys()]:
        dy = savgol_filter(
            y,
            window_length=7,
            polyorder=2,
            deriv=1,
            delta=0.01,
        )
        dy = (
            pd.DataFrame(dy)
            .rolling(min_baseline_window, center=True)
            .min()
            .interpolate(limit_direction="both")
            .values[:, 0]
        )
        dy = medfilt(dy, kernel_size=median_baseline_window)
        dy_baseline.append(dy)
    return np.concatenate(dy_baseline)


def get_step_metrics(dep_pickle_paths, target_pickle_paths):
    """Detects steps and computes related metrics.

    Parameters
    ----------
    dep_pickle_paths : pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_paths : pathlib.Path
        Paths of target pickle files to save
    """
    latency = read_pickle(dep_pickle_paths["latency"])
    xys = read_pickle(dep_pickle_paths["xys"])[:latency]
    step_events = defaultdict(list)
    for idx, hindpaw in config.STEP_MARKER_DICT.items():
        events = get_single_hindpaw_step_events(xys[:, idx, 1])
        step_events["hindpaw"] += [hindpaw] * events["num_steps"]
        step_events["t_y1"] += list(events["t_y1"])
        step_events["t_y2"] += list(events["t_y2"])
        step_events["y1"] += list(events["y1"])
        step_events["y2"] += list(events["y2"])
        step_events["t_d2y1"] += list(events["t_d2y1"])
        step_events["t_d2y2"] += list(events["t_d2y2"])
        step_events["d2y1"] += list(events["d2y1"])
        step_events["d2y2"] += list(events["d2y2"])
        step_events["t_dy_max"] += list(events["t_dy_max"])
        step_events["dy_max"] += list(events["dy_max"])
        step_events["isi_dy_max"] += list(events["isi_dy_max"])
        step_events["amp"] += list(events["y2"] - events["y1"])
        step_events["freq"] += list(
            1.0 / (np.maximum(1.0, events["isi_dy_max"])) * config.WAV_F_SAMPLING
        )
    step_events = pd.DataFrame(step_events)
    step_events["delta_phi"], step_events["abs_delta_phi"] = get_delta_phis(step_events)
    step_statistics = get_step_statistics(step_events, xys[:, 0, 1])
    step_statistics = pd.DataFrame(step_statistics)
    step_statistics["dy_baseline"] = get_hindpaw_speed_baseline(xys)
    write_pickle(step_events, target_pickle_paths["events"])
    write_pickle(step_statistics, target_pickle_paths["statistics"])


def get_step_statistics_left_right_average(step_statistics, feature_names):
    """Compute average features from both hindpaws."""
    step_statistics_left = step_statistics[step_statistics["hindpaw"] == "left"]
    step_statistics_right = step_statistics[step_statistics["hindpaw"] == "right"]
    step_statistics_paw_avg = pd.DataFrame({})
    for x_key in feature_names:
        step_statistics_paw_avg["mu_" + x_key] = (
            step_statistics_left["mu_" + x_key].values
            + step_statistics_right["mu_" + x_key].values
        ) * 0.5
        step_statistics_paw_avg["sigma_" + x_key] = (
            np.sqrt(
                step_statistics_left["sigma_" + x_key].values ** 2
                + step_statistics_right["sigma_" + x_key].values ** 2
            )
            * 0.5
        )
    return step_statistics_paw_avg


def get_steps_poses_features(latency, statistics, events, xys):
    """Computes steps and poses features."""
    features = get_step_statistics_left_right_average(
        statistics, config.STEP_FEATURES
    ).values[:latency]
    t_interpolated = np.arange(latency)
    columns = [0.0]
    for hindpaw in config.STEP_MARKER_DICT.values():
        columns[-1] += 0.5 * interp1d(
            events.query("hindpaw == @hindpaw")[config.STEP_EVENTS_TIMES].values,
            events.query("hindpaw == @hindpaw")[config.STEP_FEATURES].values,
            fill_value="extrapolate",
            kind="nearest",
            axis=0,
        )(t_interpolated)
    xys = xys[:latency]
    cm = xys[:, config.POSE_FEATURES_CM_IDX]
    i, j = config.POSE_FEATURES_HINDPAWS_IDX
    columns.append(np.abs(0.5 * (xys[:, i, 0] - xys[:, j, 0])))  # x paws
    columns.append(np.abs(0.5 * (xys[:, i, 0] + xys[:, j, 0]) - cm[:, 0, 0]))  # x paws
    columns.append(
        np.abs(xys[:, config.POSE_FEATURES_OTHER_IDX, 0] - cm[..., 0])
    )  # x other markers
    columns.append(np.abs(cm[:, 0, 0] - config.ROTAROD_WIDTH * 0.5))  # x center of mass
    columns.append(np.abs(0.5 * (xys[:, i, 1] - xys[:, j, 1])))  # y paws
    columns.append(np.abs(0.5 * (xys[:, i, 1] + xys[:, j, 1]) - cm[:, 0, 1]))  # y paws
    columns.append(
        xys[:, config.POSE_FEATURES_OTHER_IDX, 1] - cm[..., 1]
    )  # y other markers
    columns.append(cm[:, 0, 1])  # y center of mass
    features = np.column_stack([features] + columns)
    return features


def get_subsample_indices(latency, events):
    """Computes subsample indices."""
    np.random.seed(config.SUBSAMPLE_RANDOM_SEED)
    frames = (
        events[config.STEP_EVENTS_TIMES]
        .append(pd.Series([0, latency]))
        .drop_duplicates()
        .to_numpy()
    )
    frames = frames[frames <= latency]
    frames.sort()
    frames_dif = np.diff(frames)
    indices = []
    for i in range(len(frames) - 1):
        frames_between = np.random.choice(
            np.arange(frames[i], frames[i + 1]),
            size=np.minimum(config.SUBSAMPLE_BETWEEN_STEPS + 1, frames_dif[i]),
            replace=False,
        )
        frames_between.sort()
        indices.append(frames_between)
    indices = np.concatenate(indices)
    return indices


def get_steps_poses_features_subsample_indices(dep_pickle_paths, target_pickle_paths):
    """Saves steps and poses features and subsample indices.

    Parameters
    ----------
    dep_pickle_paths : pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_paths : pathlib.Path
        Paths of target pickle files to save
    """
    latency = read_pickle(dep_pickle_paths["latency"])
    statistics = read_pickle(dep_pickle_paths["statistics"])
    events = read_pickle(dep_pickle_paths["events"])
    xys = read_pickle(dep_pickle_paths["xys"])
    features = get_steps_poses_features(
        latency=latency,
        statistics=statistics,
        events=events,
        xys=xys,
    )
    indices = get_subsample_indices(
        latency=latency,
        events=events,
    )
    write_pickle(features, target_pickle_paths["stp"])
    write_pickle(indices, target_pickle_paths["idx"])


def get_standard_scaler_fit_steps_poses_features(dep_pickle_paths, target_pickle_path):
    """Fit standard scaler (Z-score), to steps and poses features."""
    scaler = StandardScaler()
    for path in dep_pickle_paths["stp"]:
        scaler.partial_fit(read_pickle(path))
    del dep_pickle_paths["stp"]
    write_pickle(scaler, target_pickle_path)


def get_pca_fit_steps_poses_features(dep_pickle_paths, target_pickle_path):
    """Fit incremental PCA to scaled steps and poses features."""
    scaler = read_pickle(dep_pickle_paths["scaler"])
    pca_stps = IncrementalPCA(copy=False)
    for path in dep_pickle_paths["stp"]:
        pca_stps.partial_fit(scaler.transform(read_pickle(path)))
    del dep_pickle_paths["stp"]
    write_pickle(pca_stps, target_pickle_path)
