"""Computes Morlet wavelet spectra of whitened joint angles."""
import numpy as np
from utilities import read_pickle, write_pickle
import config


def get_whitened_angs(angs, stats):
    angles = get_joint_angles(learning_xys[key], angle_marker_idx)
    angles -= np.mean(angles, axis=0)
    angles /= np.std(angles, axis=0)
    learning_angles[key] = angles


def dyadically_spaced_freqs(minF, maxF, numPeriods):
    """ "Get dyadically spaced frequencies."""
    minT = 1.0 / maxF
    maxT = 1.0 / minF
    Ts = minT * (
        2
        ** (
            (np.arange(numPeriods) * np.log(maxT / minT))
            / (np.log(2) * (numPeriods - 1))
        )
    )
    return (1.0 / Ts)[::-1]


def findWavelets(
    timeSeries, numComponents, freqs, numPeriods, omega0, dt, numProcessors, useGPU
):
    """
    Finds the wavelet transforms resulting from a time series.

    Parameters
        ----------
    timeSeries: array_like
        N x d array of time series values.
    numComponents: int
        Number of transforms to find.
    freqs: array_like
        Frequency channels.
    numPeriods: int
        Number of wavelet frequencies to use.
    omega0: float
        Dimensionless morlet wavelet parameter.
    dt: float
        Inverse of sampling frequency (s).
    numProcessors: int
        Number of processors to use in parallel code.
    useGPU: int
        GPU to use.
    Returns
        -------
    amplitudes: nd_array
        Wavelet amplitudes (N x (numComponents * numPeriods) )

    """
    t1 = time.time()
    print("\t Calculating wavelets, clock starting.")

    if useGPU >= 0:
        try:
            import cupy as np
        except ModuleNotFoundError as E:
            warnings.warn(
                "Trying to use GPU but cupy is not installed."
                "Install cupy or set parameters.useGPU = -1. "
                "https://docs.cupy.dev/en/stable/install.html"
            )
            raise E

        np.cuda.Device(useGPU).use()
        print("\t Using GPU #%i" % useGPU)
    else:
        import numpy as np
        import multiprocessing as mp

        if numProcessors < 0:
            numProcessors = mp.cpu_count()
        print("\t Using #%i CPUs." % numProcessors)

    timeSeries = np.array(timeSeries)
    t1 = time.time()
    N = timeSeries.shape[0]

    if useGPU >= 0:
        amplitudes = np.zeros((numPeriods * numComponents, N))
        for i in range(numComponents):
            amplitudes[
                i * numPeriods : (i + 1) * numPeriods
            ] = fastWavelet_morlet_convolution_parallel(
                i, timeSeries[:, i], freqs, omega0, dt, useGPU
            )
    else:
        try:
            pool = mp.Pool(numProcessors)
            amplitudes = pool.starmap(
                fastWavelet_morlet_convolution_parallel,
                [
                    (i, timeSeries[:, i], freqs, omega0, dt, useGPU)
                    for i in range(numComponents)
                ],
            )
            amplitudes = np.concatenate(amplitudes, 0)
            pool.close()
            pool.join()
        except Exception as E:
            pool.close()
            pool.join()
            raise E
    print("\t Done at %0.02f seconds." % (time.time() - t1))
    return amplitudes.T


def fastWavelet_morlet_convolution_parallel(modeno, x, freqs, omega0, dt, useGPU):
    """Compute wavelet amplitudes."""
    if useGPU >= 0:
        try:
            import cupy as np
        except ModuleNotFoundError as E:
            warnings.warn(
                "Trying to use GPU but cupy is not installed."
                "Install cupy or set parameters.useGPU = -1. "
                "https://docs.cupy.dev/en/stable/install.html"
            )
            raise E
    else:
        import numpy as np
    N = len(x)
    L = len(freqs)
    amp = np.zeros((L, N))

    if not N // 2:
        x = np.concatenate((x, [0]), axis=0)
        N = len(x)
        wasodd = True
    else:
        wasodd = False

    x = np.concatenate([np.zeros(int(N / 2)), x, np.zeros(int(N / 2))], axis=0)
    M = N
    N = len(x)
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * freqs)
    Omegavals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * dt)

    xHat = np.fft.fft(x)
    xHat = np.fft.fftshift(xHat)

    if wasodd:
        idx = np.arange((M / 2), (M / 2 + M - 2)).astype(int)
    else:
        idx = np.arange((M / 2), (M / 2 + M)).astype(int)

    for i in range(L):
        m = (np.pi ** (-0.25)) * np.exp(-0.5 * (-Omegavals * scales[i] - omega0) ** 2)
        q = np.fft.ifft(m * xHat) * np.sqrt(scales[i])
        q = q[idx]
        amp[i, :] = (
            np.abs(q)
            * (np.pi ** -0.25)
            * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2)
            / np.sqrt(2 * scales[i])
        )
    return amp


if __name__ == "__main__":
    samplingFreq = 100.0
    minF = 0.1
    minF_hiRes = 0.5
    maxF_hiRes = 4.0
    maxF = samplingFreq / 2.0
    numPeriods_hiRes = 20
    numPeriods_lowRes_lowF = 3
    numPeriods_lowRes_hiF = 5
    freqs = np.concatenate(
        [
            dyadically_spaced_freqs(minF, minF_hiRes, numPeriods_lowRes_lowF)[:-1],
            dyadically_spaced_freqs(minF_hiRes, maxF_hiRes, numPeriods_hiRes),
            dyadically_spaced_freqs(maxF_hiRes, maxF, numPeriods_lowRes_hiF)[1:],
        ]
    )
    numPeriods = len(freqs)
    omega0 = 10.0
    dt = 1.0 / samplingFreq
    numProcessors = -1
    useGPU = -1


def get_single_wavelet_spectrum(whitened_angs):
    pass


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
    stats = read_pickle(dep_pickle_paths["stat"])
    whitened_angs = get_whitened_angs(angs, stats)
    wavs = get_single_wavelet_spectrum(whitened_angs)
    write_pickle(wavs, target_pickle_path)
