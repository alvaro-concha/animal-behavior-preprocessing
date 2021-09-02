"""Computes Morlet wavelet spectra of whitened joint angles.

Whitens joint angles using global mean and variance.
Computes parallelized Morlet wavelet transform. Saves results.
"""
import multiprocessing as mp
import numpy as np
from utilities import read_pickle, write_pickle
import config


def get_whitened_angles(angs, mean, var):
    """
    Whitens joint angles using global mean and variance.

    Parameters
    ----------
    angs : ndarray
        Array of joint angles
    mean, var : ndarray
        Global mean and variance of the joint angles

    Returns
    -------
    whitened_angs : ndarray
        Whitened joint angles
    """
    whitened_angs = (angs - mean) / np.sqrt(var)
    return whitened_angs


def get_single_angle_wavelet_spectra(ang):
    """
    Computes Morlet wavelet amplitudes over a single whitened angle.

    Parameters
    ----------
    angs : ndarray
        Whitened joint angles

    Returns
    -------
    wav : ndarray
        Morlet wavelet spectra of the whitened angle

    Notes
    -----
    Acknowledgements to Gordon Berman's Lab MotionMapper.
    Written by Kanishk Jain (kanishkbjain@gmail.com).
    """
    len_ang = len(ang)
    wav = np.zeros((config.wav_num_channels, len_ang))

    if not len_ang // 2:
        ang = np.concatenate((ang, [0]), axis=0)
        len_ang = len(ang)
        is_len_ang_odd = True
    else:
        is_len_ang_odd = False

    ang = np.concatenate(
        [np.zeros(int(len_ang / 2)), ang, np.zeros(int(len_ang / 2))], axis=0
    )
    modified_len_ang = len_ang
    len_ang = len(ang)
    scales = (config.wav_omega_0 + np.sqrt(2 + config.wav_omega_0 ** 2)) / (
        4 * np.pi * config.wav_f_channels
    )
    omega_values = (
        2 * np.pi * np.arange(-len_ang / 2, len_ang / 2) / (len_ang * config.wav_dt)
    )

    fourier_transform = np.fft.fft(ang)
    fourier_transform = np.fft.fftshift(fourier_transform)

    if is_len_ang_odd:
        idx = np.arange(
            (modified_len_ang / 2), (modified_len_ang / 2 + modified_len_ang - 2)
        ).astype(int)
    else:
        idx = np.arange(
            (modified_len_ang / 2), (modified_len_ang / 2 + modified_len_ang)
        ).astype(int)

    for i in range(config.wav_num_channels):
        m_values = (np.pi ** (-0.25)) * np.exp(
            -0.5 * (-omega_values * scales[i] - config.wav_omega_0) ** 2
        )
        q_values = np.fft.ifft(m_values * fourier_transform) * np.sqrt(scales[i])
        q_values = q_values[idx]
        wav[i, :] = (
            np.abs(q_values)
            * (np.pi ** -0.25)
            * np.exp(
                0.25 * (config.wav_omega_0 - np.sqrt(config.wav_omega_0 ** 2 + 2)) ** 2
            )
            / np.sqrt(2 * scales[i])
        )
    return wav


def get_parallel_wavelet_spectra(angs):
    """
    Computes parallelized Morlet wavelet transform.

    Parameters
    ----------
    angs: array_like
        Whitened joint angles

    Returns
    -------
    wavs: ndarray
        Morlet wavelet amplitudes

    """
    with mp.Pool(mp.cpu_count()) as pool:
        wavs = pool.map(
            get_single_angle_wavelet_spectra,
            [angs[:, j] for j in range(angs.shape[1])],
        )
        pool.close()
        pool.join()
    wavs = np.column_stack(wavs).T
    return wavs


def get_wavelet_spectra(dep_pickle_paths, target_pickle_path):
    """
    Computes wavelet spectra of whitened joint angles.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_path : pathlib.Path
        Path of target pickle file to save
    """
    angs = read_pickle(dep_pickle_paths["ang"])
    (_, mean, var) = read_pickle(dep_pickle_paths["stat"])
    whitened_angs = get_whitened_angles(angs, mean, var)
    wavs = get_parallel_wavelet_spectra(whitened_angs)
    write_pickle(wavs, target_pickle_path)
