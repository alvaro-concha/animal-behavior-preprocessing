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
