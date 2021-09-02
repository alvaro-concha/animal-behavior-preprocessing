"""Mouse behavior analysis configuration parameters.

Control and modify all parameters in a single place.

File System
-----------
{x}_path : list of pathlib.Path
    Path to a specific folder(x)

Data Handling
-------------
{mouse, day, trial, cam}_list : list of ints
    List of mouse IDs, day, trial or camera numbers
spr_pattern : str
    Formattable string for regular expression lookup in spreadsheet names
preprocess_key_list : list of tuples
    List of keys at the beginning of preprocessing
key_list : list of tuples
    List of keys at the end of preprocessing
pickle_name : str
        Name of pickled file to save
save_folder : str
        Name of folder to save pickle into

...

Subjetct Data
-------------

Spreadsheet
-----------

Motion Tracking
---------------

Median Filter
-------------

Kalman Filter
-------------

...

"""
from pathlib import Path
from itertools import product
import numpy as np
from filterpy.common import Q_discrete_white_noise

################################# File System ##################################
################################# Config Dodo ##################################

abs_path = Path(__file__).parent.parent.parent.absolute()
spr_path = abs_path / "Data/Spreadsheets"  # Original source spreadsheets
med_path = abs_path / "Data/Positions/Median"  # xys, lhs and perspective matrices
kal_path = abs_path / "Data/Positions/Kalman"  # xys
ang_path = abs_path / "Data/Features/Angles"  # Joint angles from xys
wav_path = abs_path / "Data/Features/Wavelets"  # Wavelet spectra from joint angles
emb_path = abs_path / "Data/Embeddings"  # UMAP embeddings
met_path = abs_path / "Data/Metadata"  # Spreadsheet paths
vid_path = abs_path / "Data/Videos"  # Rotarod videos
ani_path = abs_path / "Animations"  # Output animations
fig_path = abs_path / "Figures"  # Output figures

################################ Subjetct Data #################################
################################# Config Dodo ##################################

mouse_list = [295, 297, 298, 329, 330]
# mouse_list = [295]
day_list = [1, 2, 3, 4, 5]
# day_list = [1, 5]
trial_list = [1, 2, 3, 4, 5]
# trial_list = [1]
cam_list = [1, 2]  # Front: 1, Back: 2
preprocess_key_list = list(product(mouse_list, day_list, trial_list, cam_list))
preprocess_subject_name = "M{}D{}T{}C{}"
key_list = list(product(mouse_list, day_list, trial_list))
subject_name = "M{}D{}T{}"

################################# Spreadsheet ##################################

spr_pattern = r"^(?=.*ID{})(?=.*Dia{})(?=.*trial{})(?=.*{}DLC)"
spr_skiprows = 2
spr_xy_columns = [idx for i in range(1, 49, 3) for idx in [i, i + 1]]
spr_lh_columns = range(3, 49, 3)

############################### Motion Tracking ################################

idx_marker_dict = {
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
corner_marker_idx = np.arange(12, 16)
rotarod_height = 57.0  # mm
rotarod_width = 30.0  # mm
corner_destination = np.array(
    [
        [rotarod_height, rotarod_width],
        [0.0, rotarod_width],
        [rotarod_height, 0.0],
        [0.0, 0.0],
    ]
)
front_marker_idx = np.arange(7, 12)
back_marker_idx = np.arange(7)
body_marker_idx = np.arange(12)

################################ Median Filter #################################

med_win = 11

################################ Kalman Filter #################################

kal_sigma_measurement = 0.1  # mm
kal_epsilon_lh = 1e-9  # add to avoid dividing by zero likelihood
kal_dt = 1
kal_N_ensemble = 10
kal_process_variance = 0.01  # mm squared
kal_dim_z = 1  # one coordinate at a time
kal_dim_x = 2  # 2, 3 or 4

kal_P = np.eye(kal_dim_x) * 100.0
kal_F = np.eye(kal_dim_x)
for n in range(0, kal_dim_x - kal_dim_z):
    idx = np.arange(kal_dim_x - kal_dim_z - n, dtype=int)
    kal_F[idx, (n + 1) * kal_dim_z + idx] = 1.0 / np.math.factorial(n + 1)
kal_hx = lambda x: x[0]
kal_fx = lambda x, dt: np.dot(kal_F, x)
kal_Q = Q_discrete_white_noise(
    dim=kal_dim_x,
    dt=kal_dt,
    var=kal_process_variance,
    block_size=kal_dim_z,
    order_by_dim=False,
)

################################ Joint Angles ##################################

angle_marker_idx = [
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
