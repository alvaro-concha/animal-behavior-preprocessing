"""Spreadsheet handling, median filter and perspective transform matrix.

Find, open and read motion tracking spreadsheets (CSV or Excel files).
Apply median transform to X-Y coordinates and likelihoods of body-part markers.
Compute perspective transform matrices of front and back cameras. Save results.
"""
import re
import pandas as pd
import numpy as np
from scipy.signal import medfilt
from cv2 import getPerspectiveTransform
from utilities import read_pickle, write_pickle
import config
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=2)
sns.set_style("white")

################################## FUNCTIONS ###################################


def get_spreadsheet_path(spreadsheets, key, target_pickle_path):
    """
    Saves a spreadsheet path as a pickled file.

    Parameters
    ----------
    spreadsheets : list of pathlib.Path
        Paths of all the body-part tracking spreadsheets
    key : tuple
        Data preprocessing subject key
    target_pickle_path : pathlib.Path
        Path of target pickle to save
    """
    for file in spreadsheets:
        if re.search(config.SPR_PATTERN.format(*key), file.as_posix()):
            write_pickle(file, target_pickle_path)
            return


def get_coordinates_likelihoods_from_spreadsheet_path(file):
    """
    Tries to read file as CSV or Excel, using pandas.

    Parameters
    ----------
    file : pathlib.Path
        Path to file to read

    Returns
    -------
    xys : ndarray
        X-Y coordinates from body-part tracking spreadsheet
    lhs : ndarray
        Likelihoods of body-part X-Y coordinates

    Raises
    ------
    ValueError
        If the file is not a CSV or an Excel compatible spreadsheet.
    """
    kwargs = {"skiprows": config.SPR_SKIPROWS}
    xy_kwargs = {**kwargs, "usecols": config.SPR_XY_COLUMNS}
    lh_kwargs = {**kwargs, "usecols": config.SPR_LH_COLUMNS}
    try:
        xys = pd.read_csv(file, **xy_kwargs).dropna(how="all").to_numpy()
        lhs = pd.read_csv(file, **lh_kwargs).dropna(how="all").to_numpy()
    except ValueError:
        xys = pd.read_excel(file, **xy_kwargs).dropna(how="all").to_numpy()
        lhs = pd.read_excel(file, **lh_kwargs).dropna(how="all").to_numpy()
    return xys, lhs


def get_median_filter(xys, lhs):
    """
    Applies median filter to X-Y coords and likelihoods, and reshapes coords.

    Parameters
    ----------
    xys : ndarray
        X-Y coordinates from body-part tracking spreadsheet
    lhs : ndarray
        Likelihoods of body-part X-Y coordinates

    Returns
    -------
    med_xys : ndarray
        Reshaped and filetered X-Y coordinates
    med_lhs : ndarray
        Filtered likelihoods
    """
    med_xys = medfilt(xys, kernel_size=(config.MED_WIN, 1)).reshape(
        (xys.shape[0], -1, 2)
    )
    med_lhs = medfilt(lhs, kernel_size=(config.MED_WIN, 1))
    return med_xys, med_lhs


def get_median_around_seed(xys, seed, radius):
    """
    Estimates median using values in a radius around a seed.

    Parameters
    ----------
    xys : ndarray
        X-Y coordinates of a marker over time
    seed : ndarray
        X-Y coordinates of the seed
    radius : float
        Radius of values to include around the seed
    Returns
    -------
    median : ndarray
        X-Y coordinates of the median of the marker around the seed
    """
    return np.median(xys[np.linalg.norm(xys - seed, axis=1) <= radius], axis=0)


def get_seed_radius(idx, key, mouse, day, cam):
    """Returns seed and radius based on trial information and marker index."""
    if mouse in [262, 263, 264, 265, 282] and cam == 1:
        seed = config.CORNER_SEEDS_BATCH1_CAM1[idx]
    elif mouse in [262, 263, 264, 265, 282] and cam == 2:
        seed = config.CORNER_SEEDS_BATCH1_CAM2[idx]
    elif mouse in [295, 297, 298, 329, 330]:
        seed = config.CORNER_SEEDS_BATCH2[idx]
    else:
        raise ValueError
    if (mouse, day) == (262, 3):
        radius = config.SEED_RADIUS_SENSITIVE
    elif key == (298, 1, 5, 1):
        radius = config.SEED_RADIUS_BIG
    else:
        radius = config.SEED_RADIUS
    return seed, radius


def get_rotarod_corners(key, xys):
    """
    Estimates positions of rotarod corners.

    Parameters
    ----------
    key : tuple
        Subject key (mouse, day, trial)
    xys : ndarray
        Reshaped X-Y coordinates

    Returns
    -------
    perspective_matrix : ndarray
        Perspective transform matrix
    """
    mouse, day, _, cam = key
    corners = []
    for idx in config.CORNER_MARKER_IDX:
        seed, radius = get_seed_radius(idx, key, mouse, day, cam)
        median_corner = get_median_around_seed(xys[:, idx, :], seed, radius)
        corners.append(median_corner)
    return np.array(corners)


def plot_rotarod_corners(name, key, xys, corners, path_fig):
    """Plots rotarod corners' quality control."""
    mouse, day, _, cam = key
    plt.figure(figsize=(7, 7))
    plt.scatter(*corners.T, alpha=0.9, s=100, c="k")
    for idx in config.CORNER_MARKER_IDX:
        plt.scatter(*xys[:, idx, :].T, alpha=0.1, s=1)
        seed, radius = get_seed_radius(idx, key, mouse, day, cam)
        circle = plt.Circle(seed, radius, color="grey", fill=False, alpha=0.9)
        plt.gca().add_patch(circle)
    plt.xlabel(r"$x$ (pixels)")
    plt.ylabel(r"$y$ (pixels)")
    plt.title(name)
    plt.savefig(path_fig, bbox_inches="tight")
    plt.close()


def get_perspective_matrix_corners(key, xys):
    """
    Computes perspective transform matrix.

    Parameters
    ----------
    key : tuple
        Subject key (mouse, day, trial)
    xys : ndarray
        Reshaped X-Y coordinates
    path_fig : pathlib.Path
        Path to corner quality control figure

    Returns
    -------
    perspective_matrix : ndarray
        Perspective transform matrix
    corner_source : ndarray
        Median rotarod corners
    """
    corner_source = get_rotarod_corners(key, xys)
    perspective_matrix = getPerspectiveTransform(
        np.float32(corner_source), np.float32(config.CORNER_DESTINATION)
    )
    return perspective_matrix, corner_source


def get_median_filter_perspective(name, key, dep_pickle_path, target_pickle_paths):
    """
    Applies median filter to X-Y coordinates and likelihoods.
    Reshapes X-Y coordinates. Computes perspective transform matrix.
    Saves the results as pickles.

    Parameters
    ----------
    key : tuple
        Subject key (mouse, day, trial)
    dep_pickle_path : pathlib.Path
        Path of dependency pickle file to read
    target_pickle_paths : dict of pathlib.Path
        Paths of target pickle files to save
    """
    file = read_pickle(dep_pickle_path)
    xys, lhs = get_coordinates_likelihoods_from_spreadsheet_path(file)
    xys, lhs = get_median_filter(xys, lhs)
    perspective_matrix, corners = get_perspective_matrix_corners(key, xys)
    write_pickle(xys, target_pickle_paths["med_xy"])
    write_pickle(lhs, target_pickle_paths["med_lh"])
    write_pickle(perspective_matrix, target_pickle_paths["perspective"])
    plot_rotarod_corners(name, key, xys, corners, target_pickle_paths["fig_corners"])
