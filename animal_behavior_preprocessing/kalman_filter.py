"""Kalman filter of combined front and back camera's motion tracking.

Combine front and back body-part markers, in the same perspective.
Apply parallelized Ensemble Kalman filter to combined coordinates. Saves results.
"""
import multiprocessing as mp
import numpy as np
from filterpy.kalman import EnsembleKalmanFilter
from cv2 import perspectiveTransform
from utilities import read_pickle, write_pickle
import config


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
    xys[:, config.back_marker_idx] = back_xys[:num_frames, config.back_marker_idx]
    xys[:, config.corner_marker_idx] = config.corner_destination
    lhs = front_lhs[:num_frames]
    lhs[:, config.back_marker_idx] = back_lhs[:num_frames, config.back_marker_idx]
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
        (0, config.kal_dim_x - config.kal_dim_z),
        "constant",
        constant_values=1.0,
    )
    ensemble_kalman_filter = EnsembleKalmanFilter(
        x=x_0,
        P=config.kal_P,
        dim_z=config.kal_dim_z,
        dt=config.kal_dt,
        N=config.kal_N_ensemble,
        hx=config.kal_hx,
        fx=config.kal_fx,
    )
    ensemble_kalman_filter.Q = config.kal_Q
    filtered_coordinate = []
    for coord, var in zip(coordinate, variance):
        ensemble_kalman_filter.predict()
        ensemble_kalman_filter.update(z=[coord], R=var)
        filtered_coordinate.append(config.kal_hx(ensemble_kalman_filter.x))
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
    coordinates = xys[:, config.body_marker_idx].reshape((xys.shape[0], -1))
    variances = np.repeat(lhs, 2, axis=1)
    variances = (
        config.kal_sigma_measurement / (variances + config.kal_epsilon_lh)
    ) ** 2
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            get_single_coordinate_kalman_filter,
            [(coordinates[:, j], variances[:, j]) for j in range(coordinates.shape[1])],
        )
        pool.close()
        pool.join()
    xys[:, config.body_marker_idx] = np.column_stack(results).reshape(
        (xys.shape[0], -1, 2)
    )
    return xys


def get_kalman_filter(dep_pickle_paths, target_pickle_path):
    """
    Works using median filtered and reshaped X-Y coordinates and likelihoods.
    Combines front and back cameras, using the same perspective.
    Applies Kalman filter to the result. Saves the result as pickle.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_path : pathlib.Path
        Path of target pickle file to save
    """
    xys, lhs = get_same_perspective_coordinates_likelihoods(dep_pickle_paths)
    xys = get_parallel_kalman_filter(xys, lhs)
    write_pickle(xys, target_pickle_path)
