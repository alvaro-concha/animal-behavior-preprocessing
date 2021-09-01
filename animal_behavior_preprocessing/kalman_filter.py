"""General data management functions.

Create folders, write and read pickles, get spreadsheet paths and load data.
"""
import numpy as np
from filterpy.kalman import EnsembleKalmanFilter
from filterpy.common import Q_discrete_white_noise
from cv2 import perspectiveTransform
from utilities import read_pickle, write_pickle
import config


def combine_cameras(front_xys, front_lhs, back_xys, back_lhs):
    """
    Combine same perspective front and back cameras.
    Keep shortest frame number when in conflict.

    Parameters
    ----------
    {front, back}_{xys, lhs}: ndarray
        Front or back X-Y coordinates or likelihoods

    Returns
    -------
    (xys, lhs): tuple of ndarray
        Combined coords and likelihoods
    """
    num_frames = np.minimum(len(front_xys), len(back_xys))
    xys = front_xys[:num_frames]
    xys[:, config.back_marker_idx] = back_xys[:num_frames, config.back_marker_idx]
    xys[:, config.corner_marker_idx] = config.corner_destination
    lhs = front_lhs[:num_frames]
    lhs[:, config.back_marker_idx] = back_lhs[:num_frames, config.back_marker_idx]
    return xys, lhs


def get_combined_perspective_xys_lhs(dep_pickle_paths):
    """
    Combines front and back cameras, using the same perspective.

    Parameters
    ----------
    dep_pickle_paths: dict of pathlib.Path
        Path of dependency pickle files to read

    Returns
    -------
    (xys, lhs): tuple of ndarray
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
    xys, lhs = combine_cameras(front_xys, front_lhs, back_xys, back_lhs)
    return xys, lhs


class EnKF:
    """
    Ensemble Kalman filter over measurements zs, with variance Rs.
    """

    def __init__(self, dim_z=1, dim_x=2, dt=1, N=10, var=0.01):
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.dt = dt
        self.N = N
        self.var = var
        self.P = np.eye(self.dim_x) * 100.0
        self.F = np.eye(self.dim_x)
        for n in range(0, self.dim_x - self.dim_z):
            idx = np.arange(self.dim_x - self.dim_z - n, dtype=int)
            self.F[idx, (n + 1) * self.dim_z + idx] = 1.0 / np.math.factorial(n + 1)
        self.hx = lambda x: x[0]
        self.fx = lambda x, dt: np.dot(self.F, x)
        self.Q = Q_discrete_white_noise(
            dim=self.dim_x,
            dt=self.dt,
            var=self.var,
            block_size=self.dim_z,
            order_by_dim=False,
        )

    def __call__(self, zs, Rs):
        x0 = np.pad(
            [zs[0]], (0, self.dim_x - self.dim_z), "constant", constant_values=1.0
        )
        f = EnsembleKalmanFilter(
            x=x0,
            P=self.P,
            dim_z=self.dim_z,
            dt=self.dt,
            N=self.N,
            hx=self.hx,
            fx=self.fx,
        )
        f.Q = self.Q
        filtered_signal = []
        for z, R in zip(zs, Rs):
            f.predict()
            f.update(z=[z], R=R)
            filtered_signal.append(self.hx(f.x))
        return np.array(filtered_signal)


def get_kalman_filter(dep_pickle_paths, target_pickle_path):
    """
    Works using median filtered and reshaped X-Y coordinates and likelihoods.
    Combines front and back cameras, using the same perspective.
    Applies Kalman filter to the result. Saves the result as pickle.

    Parameters
    ----------
    dep_pickle_paths: dict of pathlib.Path
        Path of dependency pickle files to read
    target_pickle_path: pathlib.Path
        Paths of target pickle file to save
    """
    xys, lhs = get_combined_perspective_xys_lhs(dep_pickle_paths)
    kalman_filter = EnKF(
        dim_z=dim_z, dim_x=dim_x, dt=dt, N=N_ensemble, var=process_variance
    )

    write_pickle(xys, target_pickle_path)
