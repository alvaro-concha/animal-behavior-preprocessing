"""Kalman filter of combined front and back camera's motion tracking.

Combine front and back body-part markers, in the same perspective.
Apply parallelized Ensemble Kalman filter to combined coordinates. Save results.
"""
import multiprocessing as mp
import numpy as np
from scipy.stats import iqr
from filterpy.kalman import EnsembleKalmanFilter
from cv2 import perspectiveTransform
from utilities import read_pickle, write_pickle
import config
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=2)
sns.set_style("white")


def get_combined_cameras_coordinates_likelihoods(
    front_xys, front_lhs, back_xys, back_lhs
):
    """
    Combine same perspective front and back cameras.
    Keep shortest frame number when in conflict.

    Parameters
    ----------
    {front, back}_{xys, lhs} : ndarray
        Front or back X-Y coordinates or likelihoods

    Returns
    -------
    (xys, lhs) : tuple of ndarray
        Combined coords and likelihoods
    """
    num_frames = np.minimum(len(front_xys), len(back_xys))
    xys = front_xys[:num_frames]
    xys[:, config.BACK_MARKER_IDX] = back_xys[:num_frames, config.BACK_MARKER_IDX]
    xys[:, config.CORNER_MARKER_IDX] = config.CORNER_DESTINATION
    lhs = front_lhs[:num_frames]
    lhs[:, config.BACK_MARKER_IDX] = back_lhs[:num_frames, config.BACK_MARKER_IDX]
    return xys, lhs


def get_same_perspective_coordinates_likelihoods(dep_pickle_paths):
    """
    Combines front and back cameras, using the same perspective.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Path of dependency pickle files to read

    Returns
    -------
    (xys, lhs) : tuple of ndarray
        Combined coords and likelihoods
    """
    front_xys = read_pickle(dep_pickle_paths["front_med_xy"])
    front_lhs = read_pickle(dep_pickle_paths["front_med_lh"])
    front_perspective_matrix = read_pickle(dep_pickle_paths["front_perspective"])
    back_xys = read_pickle(dep_pickle_paths["back_med_xy"])
    back_lhs = read_pickle(dep_pickle_paths["back_med_lh"])
    back_perspective_matrix = read_pickle(dep_pickle_paths["back_perspective"])
    front_xys = perspectiveTransform(front_xys, front_perspective_matrix)
    back_xys = perspectiveTransform(back_xys, back_perspective_matrix)
    xys, lhs = get_combined_cameras_coordinates_likelihoods(
        front_xys, front_lhs, back_xys, back_lhs
    )
    return xys, lhs


def get_single_coordinate_kalman_filter(coordinate, variance):
    """
    Ensemble Kalman filter over a single coordinate.

    Parameters
    ----------
    coordinate : ndarray
        Single coordinate time series to filter

    variance : ndarray
        Variance of this coordinate time series

    Returns
    -------
    filtered_coordinate : ndarray
        Filtered single coordinate time series
    """
    x_0 = np.pad(
        [coordinate[0]],
        (0, config.KAL_DIM_X - config.KAL_DIM_Z),
        "constant",
        constant_values=1.0,
    )
    ensemble_kalman_filter = EnsembleKalmanFilter(
        x=x_0,
        P=config.KAL_P,
        dim_z=config.KAL_DIM_Z,
        dt=config.KAL_DT,
        N=config.KAL_N_ENSEMBLE,
        hx=config.KAL_HX,
        fx=config.KAL_FX,
    )
    ensemble_kalman_filter.Q = config.KAL_Q
    filtered_coordinate = []
    for coord, var in zip(coordinate, variance):
        ensemble_kalman_filter.predict()
        ensemble_kalman_filter.update(z=[coord], R=var)
        filtered_coordinate.append(config.KAL_HX(ensemble_kalman_filter.x))
    return np.array(filtered_coordinate)


def get_parallel_kalman_filter(xys, lhs):
    """
    Apply parallelized Kalman filter.

    Parameters
    ----------
    xys : ndarray
        Combined X-Y coordinates from both cameras in same perspective
    lhs : ndarray
        Combined likelihoods

    Returns
    -------
    xys : ndarray
        Kalman filtered X-Y coordinates
    """
    coordinates = xys[:, config.BODY_MARKER_IDX].reshape((xys.shape[0], -1))
    variances = np.repeat(lhs, 2, axis=1)
    variances = (
        config.KAL_SIGMA_MEASUREMENT / (variances + config.KAL_EPSILON_LH)
    ) ** 2
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            get_single_coordinate_kalman_filter,
            [(coordinates[:, j], variances[:, j]) for j in range(coordinates.shape[1])],
        )
        pool.close()
        pool.join()
    xys[:, config.BODY_MARKER_IDX] = np.column_stack(results).reshape(
        (xys.shape[0], -1, 2)
    )
    return xys


def plot_kalman_filter(name, xys, path_fig_x, path_fig_y):
    """Plots Kalman filter markers' quality control."""
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


def get_kalman_filter(name, dep_pickle_paths, target_pickle_paths):
    """
    Works using median filtered and reshaped X-Y coordinates and likelihoods.
    Combines front and back cameras, using the same perspective.
    Applies Kalman filter to the result. Saves the result as pickle.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_paths : pathlib.Path
        Paths of target pickle files to save
    """
    xys, lhs = get_same_perspective_coordinates_likelihoods(dep_pickle_paths)
    xys = get_parallel_kalman_filter(xys, lhs)
    write_pickle(xys, target_pickle_paths["kal_xy"])
    plot_kalman_filter(
        name,
        xys[:, config.BODY_MARKER_IDX, :],
        target_pickle_paths["fig_kal_x"],
        target_pickle_paths["fig_kal_y"],
    )
