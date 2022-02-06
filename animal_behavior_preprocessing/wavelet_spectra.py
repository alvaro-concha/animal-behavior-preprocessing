"""Computes Morlet wavelet spectra of whitened joint angles.

Whitens joint angles using global mean and variance.
Computes parallelized Morlet wavelet transform. Save results.
"""
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import IncrementalPCA
from utilities import read_pickle, write_pickle
import config


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
    Inspired by Kanishk Jain (kanishkbjain@gmail.com).
    """
    len_ang = len(ang)
    wav = np.zeros((config.WAV_NUM_CHANNELS, len_ang))

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
    scales = (config.WAV_OMEGA_0 + np.sqrt(2 + config.WAV_OMEGA_0 ** 2)) / (
        4 * np.pi * config.WAV_F_CHANNELS
    )
    omega_values = (
        2 * np.pi * np.arange(-len_ang / 2, len_ang / 2) / (len_ang * config.WAV_DT)
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

    for i in range(config.WAV_NUM_CHANNELS):
        m_values = (np.pi ** (-0.25)) * np.exp(
            -0.5 * (-omega_values * scales[i] - config.WAV_OMEGA_0) ** 2
        )
        q_values = np.fft.ifft(m_values * fourier_transform) * np.sqrt(scales[i])
        q_values = q_values[idx]
        wav[i, :] = (
            np.abs(q_values)
            * (np.pi ** -0.25)
            * np.exp(
                0.25 * (config.WAV_OMEGA_0 - np.sqrt(config.WAV_OMEGA_0 ** 2 + 2)) ** 2
            )
            / np.sqrt(2 * scales[i])
        )
    return wav.T


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
    wavs = np.column_stack(wavs)
    return wavs


def get_wavelet_spectra(path_scaler_ang, path_ang, target_pickle_path):
    """
    Computes wavelet spectra of whitened joint angles.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_path : pathlib.Path
        Path of target pickle file to save
    """
    scaler_angs = read_pickle(path_scaler_ang)
    angs = read_pickle(path_ang)
    print("Shape of read joint angles:", angs.shape)
    angs = scaler_angs.transform(angs)
    wavs = get_parallel_wavelet_spectra(angs)
    print("Shape of created wavelet spectra:", wavs.shape)
    write_pickle(wavs, target_pickle_path)


def get_pca_fit_wavelet_spectra(dep_pickle_paths, target_pickle_path):
    """Fit incremental PCA to wavelet spectra features."""
    pca_wavs = IncrementalPCA(copy=False)
    for path in dep_pickle_paths["wav"]:
        pca_wavs.partial_fit(read_pickle(path))
    del dep_pickle_paths["wav"]
    write_pickle(pca_wavs, target_pickle_path)
