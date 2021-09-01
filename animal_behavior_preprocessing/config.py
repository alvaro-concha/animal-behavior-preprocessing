"""Mouse behavior analysis configuration parameters.

Control and change all parameters in a single place.

File System
-----------
{}_path: list of pathlib.Path
    Path to a specific folder

Data Handling
-------------
{mouse, day, trial, cam}_list: list of ints
    List of mouse IDs, day, trial or camera numbers
spr_pattern: str
    Formattable string for regular expression lookup in spreadsheet names
preprocess_key_list: list of tuples
    List of keys at the beginning of preprocessing
key_list: list of tuples
    List of keys at the end of preprocessing
pickle_name: str
        Name of pickled file to save
save_folder: str
        Name of folder to save pickle into
"""
from pathlib import Path
from itertools import product
import numpy as np

################################# File System ##################################

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

# mouse_list = [295, 297, 298, 329, 330]
mouse_list = [295]
day_list = [1, 5]
# trial_list = [1, 2, 3, 4, 5]
trial_list = [1]
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
    14: "down right",
    15: "down left",
}
corner_marker_idx = np.arange(12, 16)
corner_destination = np.array([[5.7, 3], [0, 3], [5.7, 0], [0, 0]]) * 10.0  # in mm
front_marker_idx = np.arange(7, 12)
back_marker_idx = np.arange(7)

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
