"""Quantile filter and smoother of Kalman filter outputs.

Apply parallelized quantile filter and smoother to Kalman filter outputs. Save results.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.stats import iqr, trim_mean
from scipy.signal import medfilt
from utilities import read_pickle, write_pickle
import config
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=2)
sns.set_style("white")


def get_window_trim_mean(coordinate, window, cut):
    """
    Apply trimmed mean over a window.
    """
    coordinate_window = np.column_stack(
        [coordinate[i : i + window] for i in range(len(coordinate) - window + 1)]
    ).T
    coordinate[window // 2 : -window // 2 + 1] = trim_mean(
        a=coordinate_window, proportiontocut=cut, axis=1
    )
    return np.array(coordinate)


def get_single_coordinate_quantile_filter_smoother(coordinate, idx, is_y):
    """
    Quantile filter over a single coordinate.

    Parameters
    ----------
    coordinate : ndarray
        Single coordinate time series to filter
    idx : int
        Marker index
    is_y : int
        Value 1 if coordinate is Y, value 0 if coordinate is X

    Returns
    -------
    coordinate : ndarray
        Filtered single coordinate time series
    """
    if (not is_y) and (idx in config.QNT_X_SWITCH_MARKER_IDX):
        qnt_keep = config.QNT_X_SWITCH_KEEP
    else:
        qnt_keep = config.QNT_KEEP
    qnt_leave = 1.0 - qnt_keep
    qnt_values = (qnt_leave * 0.5, 0.5, 1 - qnt_leave * 0.5)
    rolling_quantiles = np.column_stack(
        [
            pd.DataFrame(coordinate)
            .rolling(config.QNT_WIN, center=True)
            .quantile(q)
            .interpolate(limit_direction="both")
            .values[:, 0]
            for q in qnt_values
        ]
    )
    median = rolling_quantiles[:, 1, np.newaxis]
    rolling_quantiles = (rolling_quantiles - median) * config.QNT_EXPANSION + median
    coordinate = np.where(
        coordinate <= rolling_quantiles[:, 0],
        rolling_quantiles[:, 0],
        coordinate,
    )
    coordinate = np.where(
        coordinate >= rolling_quantiles[:, 2],
        rolling_quantiles[:, 2],
        coordinate,
    )
    coordinate = np.where(
        np.abs(np.arange(len(coordinate)) - len(coordinate) // 2)
        <= config.QNT_TRIM_WIN // 2,
        medfilt(coordinate, config.QNT_TRIM_WIN),
        coordinate,
    )
    coordinate = get_window_trim_mean(
        coordinate=coordinate,
        window=config.QNT_TRIM_WIN,
        cut=config.QNT_TRIM_CUT,
    )
    return coordinate


def get_parallel_quantile_filter_smoother(xys):
    """
    Apply parallelized quantile filter.

    Parameters
    ----------
    xys : ndarray
        Combined X-Y coordinates from both cameras in same perspective
    lhs : ndarray
        Combined likelihoods

    Returns
    -------
    xys : ndarray
        Quantile filtered X-Y coordinates
    """
    coordinates = xys[:, config.BODY_MARKER_IDX].reshape((xys.shape[0], -1))
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            get_single_coordinate_quantile_filter_smoother,
            [
                (coordinates[:, j], config.BODY_MARKER_IDX[j // 2], j % 2)
                for j in range(coordinates.shape[1])
            ],
        )
        pool.close()
        pool.join()
    xys[:, config.BODY_MARKER_IDX] = np.column_stack(results).reshape(
        (xys.shape[0], -1, 2)
    )
    return xys


def plot_quantile_filter_smoother(name, xys, path_fig_x, path_fig_y):
    """Plots quantile filter markers' quality control."""
    scale = iqr(xys, axis=0, nan_policy="omit", keepdims=True)
    loc = np.median(xys, axis=0, keepdims=True)
    xys = 0.5 * (xys - loc) / scale
    time = np.arange(len(xys)) / 60.0 / config.WAV_F_SAMPLING
    plt.figure(figsize=(50, 7))
    for idx in range(xys.shape[1]):
        plt.plot(time, xys[:, idx, 0] + idx, alpha=0.9)
    plt.xlabel(r"$t$ (min)")
    plt.ylabel(r"$x$")
    plt.ylim(-2, xys.shape[1] + 1)
    plt.title(name)
    plt.savefig(path_fig_x, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(50, 7))
    for idx in range(xys.shape[1]):
        plt.plot(time, xys[:, idx, 1] + idx, alpha=0.9)
    plt.xlabel(r"$t$ (min)")
    plt.ylabel(r"$y$")
    plt.ylim(-2, xys.shape[1] + 1)
    plt.title(name)
    plt.savefig(path_fig_y, bbox_inches="tight")
    plt.close()


def get_smoothed_nose_center_of_mass(xys):
    """
    Adds a smoothed nose and center of mass coordinates. Saves the result as pickle.
    """
    nose = np.apply_along_axis(
        func1d=get_window_trim_mean,
        axis=0,
        arr=xys[:, config.NOSE_MARKER_IDX],
        window=config.SMOOTH_TRIM_WIN,
        cut=config.SMOOTH_TRIM_CUT,
    )
    cm = np.apply_along_axis(
        func1d=get_window_trim_mean,
        axis=0,
        arr=xys[:, config.CM_MARKER_IDX].mean(axis=1, keepdims=True),
        window=config.SMOOTH_TRIM_WIN,
        cut=config.SMOOTH_TRIM_CUT,
    )
    return np.column_stack([xys, nose, cm])


def get_quantile_filter_smoother(name, dep_pickle_path, target_pickle_paths):
    """
    Applies parallelized quantile filter and smoother. Saves the result as pickle.

    Parameters
    ----------
    dep_pickle_path : dict of pathlib.Path
        Path of dependency pickle file to read
    target_pickle_paths : pathlib.Path
        Paths of target pickle files to save
    """
    xys = read_pickle(dep_pickle_path)
    xys = get_parallel_quantile_filter_smoother(xys)
    xys = get_smoothed_nose_center_of_mass(xys)
    write_pickle(xys, target_pickle_paths["qnt_xy"])
    plot_quantile_filter_smoother(
        name,
        xys[:, np.concatenate([config.BODY_MARKER_IDX, config.EXTRA_MARKER_IDX]), :],
        target_pickle_paths["fig_qnt_x"],
        target_pickle_paths["fig_qnt_y"],
    )
