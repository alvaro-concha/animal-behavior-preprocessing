"""Mouse behavior preprocessing modeules configuration parameters.

Control and modify parameters in the modules of the pipeline.

Spreadsheet
-----------
SPR_PATTERN : str
    Formattable string: regular expression lookup in spreadsheet names
SPR_SKIPROWS : int
    Number of rows to skip when reading spreadsheet
SPR_XY_COLUMNS : array_like
    Indices of columns containing X-Y coordinates
SPR_LH_COLUMNS : array_like
    Indices of columns containing likelihoods

Motion Tracking
---------------
IDX_MARKER_DICT : dict
    Dictionary of {int: str}: mapping marker indices to their names
{FRONT, BACK}_MARKER_IDX : array_like
    Body-part marker indices appearing clearly in front or back camera
{BODY, CORNER}_MARKER_IDX : array_like
    Body-part or rotarod corner marker indices
ROTAROD_{HEIGHT, WIDTH} : float
    Rotarod height and width in mm, modelling it as a rectangle
CORNER_DESTINATION : array_like
    Ideal locations of the four rotarod corners. Coordinates reference frame

Median Filter
-------------
MED_WIN : int
    Odd integer: median filter window size

Kalman Filter
-------------
KAL_SIGMA_MEASUREMENT : float
    Kalman filter external meauserement error
KAL_EPSILON_LH : float
    Tiny number: avoid dividing by zero when converting likelihoods to errors
KAL_DT : float
    Kalman filter time step size
KAL_N_ENSEMBLE : int
    Size of ensemble
KAL_PROCESS_VARIANCE : float
    Kalman filter internal process variance
KAL_DIM_Z : int
    Dimension of external meauserement (set to 1 to use parallel Kalman filter)
KAL_DIM_X : int
    Dimension of the internal model of the Kalman filter
KAL_P : array_like
    Initial covariance matrix of the internal model of the Kalman filter
KAL_F : array_like
    Transition matrix of the internal model of the Kalman filter
KAL_HX : function
    Measurement function
KAL_FX : function
    Transition function
KAL_Q : array_like
    Kalman filter internal process covariance matrix

Joint Angles
------------
ANGLE_MARKER_IDX : array_like
    Array of thruples of marker indices that define an angle

Wavelet Spectra
---------------
WAV_F_SAMPLING : float
    Sampling frequency of motion tracking, in Hz
WAV_F_MIN : float
    Minimum frequency channel value, in Hz
WAV_F_MIN_MID : float
    Minimum frequency channel value, in higher resolution midband, in Hz
WAV_F_MAX_MID : float
    Maximum frequency channel value, in higher resolution midband, in Hz
WAV_F_MAX : float
    Maximum frequency channel value, in Hz. Set to Nyquist frequency
WAV_NUM_CHANNELS_{LOW, MID, HIGH} : int
    Number of frequency channels in low, mid and high frequency bands
WAV_F_CHANNELS : array_like
    Step-wise generated, dyadically spaced total frequency channels
WAV_NUM_CHANNELS : int
    Number of total frequency channels
WAV_OMEGA_0 : int
    Morlet wavelet omega_0 parameter. Related to number of observed cycles
WAV_DT : float
    Wavelet transform time step value. Set to inverse of sampling frequency
"""
import numpy as np
from filterpy.common import Q_discrete_white_noise
from dyadic_frequencies import get_dyadic_frequencies

################################# Spreadsheet ##################################

SPR_PATTERN = r"^(?=.*ID{})(?=.*Dia{})(?=.*trial{})(?=.*{}DLC)"
SPR_SKIPROWS = 2
SPR_XY_COLUMNS = [idx for i in range(1, 49, 3) for idx in [i, i + 1]]
SPR_LH_COLUMNS = range(3, 49, 3)

############################### Motion Tracking ################################

IDX_MARKER_DICT = {
    0: "left hind leg",
    1: "right hind leg",
    2: "base tail",
    3: "middle tail",
    4: "back",
    5: "left back",
    6: "right back",
    7: "left front leg",
    8: "right front leg",
    9: "nose",
    10: "left ear",
    11: "right ear",
    12: "top right",
    13: "top left",
    14: "bottom right",
    15: "bottom left",
}
FRONT_MARKER_IDX = np.arange(7, 12)
BACK_MARKER_IDX = np.arange(7)
BODY_MARKER_IDX = np.arange(12)
CORNER_MARKER_IDX = np.arange(12, 16)
ROTAROD_HEIGHT = 57.0  # in mm
ROTAROD_WIDTH = 30.0  # in mm
CORNER_DESTINATION = np.array(
    [
        [ROTAROD_HEIGHT, ROTAROD_WIDTH],
        [0.0, ROTAROD_WIDTH],
        [ROTAROD_HEIGHT, 0.0],
        [0.0, 0.0],
    ]
)

################################ Median Filter #################################

MED_WIN = 11

################################ Kalman Filter #################################

KAL_SIGMA_MEASUREMENT = 0.1  # mm
KAL_EPSILON_LH = 1e-9  # add to avoid dividing by zero likelihood
KAL_DT = 1
KAL_N_ENSEMBLE = 10
KAL_PROCESS_VARIANCE = 0.01  # mm squared
KAL_DIM_Z = 1  # one coordinate at a time
KAL_DIM_X = 2  # 2, 3 or 4
KAL_P = np.eye(KAL_DIM_X) * 100.0
KAL_F = np.eye(KAL_DIM_X)
for n in range(0, KAL_DIM_X - KAL_DIM_Z):
    idx = np.arange(KAL_DIM_X - KAL_DIM_Z - n, dtype=int)
    KAL_F[idx, (n + 1) * KAL_DIM_Z + idx] = 1.0 / np.math.factorial(n + 1)
KAL_HX = lambda x: x[0]
KAL_FX = lambda x, dt: np.dot(KAL_F, x)
KAL_Q = Q_discrete_white_noise(
    dim=KAL_DIM_X,
    dt=KAL_DT,
    var=KAL_PROCESS_VARIANCE,
    block_size=KAL_DIM_Z,
    order_by_dim=False,
)

################################ Joint Angles ##################################

ANGLE_MARKER_IDX = [
    [7, 4, 8],
    [4, 8, 7],
    [8, 7, 4],
    [14, 9, 15],
    [9, 15, 14],
    [15, 14, 9],
    [10, 9, 11],
    [9, 11, 4],
    [11, 4, 10],
    [4, 10, 9],
    [16, 9, 4],
    [9, 4, 16],
    [4, 16, 9],
    [2, 0, 5],
    [0, 5, 4],
    [5, 4, 6],
    [4, 6, 1],
    [6, 1, 2],
    [1, 2, 0],
    [4, 3, 2],
    [2, 4, 3],
    [3, 2, 4],
    [2, 4, 16],
    [4, 16, 2],
    [16, 2, 4],
    [14, 2, 15],
    [2, 15, 14],
    [15, 14, 2],
]

############################### Wavelet Spectra ################################

WAV_F_SAMPLING = 100.0
WAV_F_MIN = 0.1
WAV_F_MIN_MID = 0.5
WAV_F_MAX_MID = 4.0
WAV_F_MAX = WAV_F_SAMPLING / 2.0
WAV_NUM_CHANNELS_LOW = 3
WAV_NUM_CHANNELS_MID = 20
WAV_NUM_CHANNELS_HIGH = 5
WAV_F_CHANNELS = np.concatenate(
    [
        get_dyadic_frequencies(WAV_F_MIN, WAV_F_MIN_MID, WAV_NUM_CHANNELS_LOW)[:-1],
        get_dyadic_frequencies(WAV_F_MIN_MID, WAV_F_MAX_MID, WAV_NUM_CHANNELS_MID),
        get_dyadic_frequencies(WAV_F_MAX_MID, WAV_F_MAX, WAV_NUM_CHANNELS_HIGH)[1:],
    ]
)
WAV_NUM_CHANNELS = len(WAV_F_CHANNELS)
WAV_OMEGA_0 = 10.0
WAV_DT = 1.0 / WAV_F_SAMPLING
