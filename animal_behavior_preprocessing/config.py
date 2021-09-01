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

################################# File System ##################################

abs_path = Path(__file__).parent.parent.parent.absolute()
spr_path = abs_path / "Data/Spreadsheets"
pos_path = abs_path / "Data/Positions"
med_path = abs_path / "Data/Positions/Median"
# per_path = abs_path / "Data/Positions/Perspective" Podria meter a Perspective mezclado dentro de median
kal_path = abs_path / "Data/Positions/Kalman"
ang_path = abs_path / "Data/Features/Angles"
wav_path = abs_path / "Data/Features/Wavelets"
emb_path = abs_path / "Data/Embeddings"
met_path = abs_path / "Data/Metadata"
ani_path = abs_path / "Animations"
fig_path = abs_path / "Figures"

################################ Data Handling #################################

# mouse_list = [295, 297, 298, 329, 330]
mouse_list = [295]
day_list = [1, 5]
# trial_list = [1, 2, 3, 4, 5]
trial_list = [1]
cam_list = [1, 2]

preprocess_key_list = list(product(mouse_list, day_list, trial_list, cam_list))
preprocess_subject_name = "M{}D{}T{}C{}"
key_list = list(product(mouse_list, day_list, trial_list))
subject_name = "M{}D{}T{}"
spr_pattern = r"^(?=.*ID{})(?=.*Dia{})(?=.*trial{})(?=.*{}DLC)"
spr_skiprows = 2
spr_xy_columns = [idx for i in range(1, 49, 3) for idx in [i, i + 1]]
spr_lh_columns = range(3, 49, 3)
