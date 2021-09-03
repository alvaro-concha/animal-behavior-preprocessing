"""Spreadsheet handling, median filter and perspective transform matrix.

Find, open and read motion tracking spreadsheets (CSV or Excel files).
Apply median transform to X-Y coordinates and likelihoods of body-part markers.
Compute perspective transform matrices of front and back cameras. Saves results.
"""
import re
import pandas as pd
import numpy as np
from scipy.signal import medfilt
from cv2 import getPerspectiveTransform
from utilities import read_pickle, write_pickle
import config

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


def get_perspective_matrix(xys):
    """
    Computes perspective transform matrix.

    Parameters
    ----------
    xys : ndarray
        Reshaped X-Y coordinates

    Returns
    -------
    perspective_matrix : ndarray
        Perspective transform matrix
    """
    corner_source = np.median(xys[:, config.CORNER_MARKER_IDX, :], axis=0)
    perspective_matrix = getPerspectiveTransform(
        np.float32(corner_source), np.float32(config.CORNER_DESTINATION)
    )
    return perspective_matrix


def get_median_filter_perspective(dep_pickle_path, target_pickle_paths):
    """
    Applies median filter to X-Y coordinates and likelihoods.
    Reshapes X-Y coordinates. Computes perspective transform matrix.
    Saves the results as pickles.

    Parameters
    ----------
    dep_pickle_path : pathlib.Path
        Path of dependency pickle file to read
    target_pickle_paths : dict of pathlib.Path
        Paths of target pickle files to save
    """
    file = read_pickle(dep_pickle_path)
    xys, lhs = get_coordinates_likelihoods_from_spreadsheet_path(file)
    xys, lhs = get_median_filter(xys, lhs)
    perspective_matrix = get_perspective_matrix(xys)
    write_pickle(xys, target_pickle_paths["med_xy"])
    write_pickle(lhs, target_pickle_paths["med_lh"])
    write_pickle(perspective_matrix, target_pickle_paths["perspective"])
