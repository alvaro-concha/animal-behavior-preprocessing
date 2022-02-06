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
ROTAROD_{WIDTH, HEIGHT} : float
    Rotarod height and width in mm, as seen in the video
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

Quantile Filter
---------------
QNT_KEEP : float
    Proportion (between 0 and 1) to keep when filtering
QNT_X_SWITCH_KEEP : float
    Proportion to keep when filtering markers that suffer horizontal switching
QNT_X_SWITCH_MARKER_IDX : list
    List of marker indices that suffer from horizontal switching
QNT_EXPANSION : float 
    Factor to increase quantile bands
QNT_WIN : int
    Odd integer: quantile filter window size
QNT_TRIM_WIN : int
    Odd integer: quantile filter window size for trimming
QNT_TRIM_CUT : float
    Proportion (between 0 and 1) quantile bands to trim

Smoothed Nose and Center of Mass
--------------------------------
SMOOTH_TRIM_WIN : int
    Odd integer: nose and center window size for trimming
SMOOTH_TRIM_CUT : float
    Proportion (between 0 and 1) quantile bands to trim

Step Metrics
------------
STEP_MARKER_DICT : dict
    Dictionary of {int: str}: mapping of hindpaw marker indices to their names

Steps and Poses
---------------
STEP_EVENTS_TIMES : str
    Name of the column containing step events times
STEP_FEATURES : list
    List of features to extract from step events
POSE_FEATURES_CM_IDX : int
    Index of the column containing the center of mass
POSE_FEATURES_HINDPAWS_IDX : list
    List of indices of the columns containing the hindpaw marker coordinates
POSE_FEATURES_OTHER_IDX : list
    List of indices of the columns containing other marker coordinates

Subsample Indices
-----------------
SUBSAMPLE_RANDOM_SEED : int
    Random seed for subsampling
SUBSAMPLE_BETWEEN_STEPS : int
    Number of frames to cover between steps in the subsampling

Joint Angles
------------
ANG_MARKER_IDX : array_like
    Array of thruples of marker indices that define an angle
ANG_TRIM_WIN : int
    Odd integer: angle window size for trimming
ANG_TRIM_CUT : float
    Proportion (between 0 and 1) quantile bands to trim

Wavelet Spectra
---------------
WAV_F_SAMPLING : float
    Sampling frequency of motion tracking, in Hz
WAV_F_MIN : float
    Minimum frequency channel value, in Hz
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
    0: "left hind paw",
    1: "right hind paw",
    2: "base tail",
    3: "middle tail",
    4: "back",
    5: "left back",
    6: "right back",
    7: "left front paw",
    8: "right front paw",
    9: "nose",
    10: "left ear",
    11: "right ear",
    12: "top right",
    13: "top left",
    14: "bottom right",
    15: "bottom left",
    16: "smoothed nose",
    17: "center of mass",
}
FRONT_MARKER_IDX = np.arange(7, 12)
BACK_MARKER_IDX = np.arange(7)
BODY_MARKER_IDX = np.arange(12)
CORNER_MARKER_IDX = np.arange(12, 16)
EXTRA_MARKER_IDX = np.arange(16, 18)
CM_MARKER_IDX = [2, 4, 5, 6]
NOSE_MARKER_IDX = [9]
ROTAROD_WIDTH = 57.0  # in mm
ROTAROD_HEIGHT = 30.0  # in mm
CORNER_DESTINATION = np.array(
    [
        [ROTAROD_WIDTH, ROTAROD_HEIGHT],
        [0.0, ROTAROD_HEIGHT],
        [ROTAROD_WIDTH, 0.0],
        [0.0, 0.0],
    ]
)
CORNER_SEEDS_BATCH1_CAM1 = {
    12: [500, 200],
    13: [100, 200],
    14: [500, 450],
    15: [100, 450],
}
CORNER_SEEDS_BATCH1_CAM2 = {
    12: [425, 150],
    13: [25, 150],
    14: [425, 450],
    15: [25, 450],
}
CORNER_SEEDS_BATCH2 = {
    12: [525, 140],
    13: [125, 140],
    14: [525, 440],
    15: [125, 440],
}
SEED_RADIUS = 200
SEED_RADIUS_SENSITIVE = 150
SEED_RADIUS_BIG = 250

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

############################### Quantile Filter ################################

QNT_KEEP = 0.75
QNT_X_SWITCH_KEEP = 0.5
QNT_X_SWITCH_MARKER_IDX = [0, 1, 5, 6, 7, 8, 10, 11]
QNT_EXPANSION = 2.5
QNT_WIN = 3001
QNT_TRIM_WIN = 7
QNT_TRIM_CUT = 0.25

###################### Smoothed Nose and Center of Mass ########################

SMOOTH_TRIM_WIN = 101
SMOOTH_TRIM_CUT = 0.1

################################ Step Metrics ##################################

STEP_MARKER_DICT = {
    0: "left",  # left hind paw
    1: "right",  # right hind paw
}

############################## Steps and Poses #################################

STEP_EVENTS_TIMES = "t_dy_max"
STEP_FEATURES = ["y1", "y2", "amp", "dy_max", "abs_delta_phi", "freq"]
POSE_FEATURES_CM_IDX = [17]
POSE_FEATURES_HINDPAWS_IDX = [0, 1]
POSE_FEATURES_OTHER_IDX = [2, 4, 16]

############################# Subsample Indices ################################

SUBSAMPLE_RANDOM_SEED = 42
SUBSAMPLE_BETWEEN_STEPS = 2  # covers around 7 to 11% of each trial

################################ Joint Angles ##################################

ANG_MARKER_IDX = [
    [4, 17, 0],  # left hindpaw
    [17, 0, 4],  # left hindpaw
    [4, 17, 1],  # right hindpaw
    [17, 1, 4],  # right hindpaw
    [0, 17, 1],  # cm hindpaws
    [17, 2, 3],  # tail
    [2, 17, 4],  # back
    [17, 4, 2],  # back
    [4, 16, 17],  # nose
    [17, 4, 16],  # nose
]
ANG_TRIM_WIN = 9
ANG_TRIM_CUT = 0.25

############################### Wavelet Spectra ################################

WAV_F_SAMPLING = 100.0
WAV_F_MIN = 0.1
WAV_F_MAX = 10.0
WAV_NUM_CHANNELS = 50
WAV_F_CHANNELS = get_dyadic_frequencies(WAV_F_MIN, WAV_F_MAX, WAV_NUM_CHANNELS)
WAV_OMEGA_0 = 10.0
WAV_DT = 1.0 / WAV_F_SAMPLING
