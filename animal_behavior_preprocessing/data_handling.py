"""General data management functions.

Create folders, write and read pickles, get spreadsheet paths and load data.
"""
import pandas as pd
import pickle
import config
import re

################################## FUNCTIONS ###################################


def write_pickle(obj, file, path):
    """
    Writes an object as a pickle file.

    Parameters
    ----------
    file: str
        Name of the pickle to be created
    path: pathlib.Path
        Path to save the pickle in
    obj: pickable object
        Object to be pickled
    """
    path.mkdir(parents=True, exist_ok=True)
    pickle_path = path / file
    with open(pickle_path, "wb") as file:
        pickle.dump(obj, file, protocol=-1, fix_imports=False)


def read_pickle(file, path):
    """
    Loads an object from a pickle file

    Parameters
    ----------
    file: str
        Name of the pickle to be created
    path: pathlib.Path
        Path to save the pickle in

    Returns
    -------
    obj: pickable object
        Unpickled object
    """
    pickle_path = path / file
    with open(pickle_path, "rb") as pickled_object:
        return pickle.load(pickled_object)


def get_spr_path(spreadsheets, key, pickle_name):
    """
    Saves ordered spreadsheet paths as a pickled dict
    spr_paths_dict[preprocess_key] = spr_path.

    Returns
    -------
    spr_paths_dict: dict of pathlib.Path with tuple keys
        Ordered spreadsheet paths
    """
    for file in spreadsheets:
        if re.search(config.spr_pattern.format(*key), file.as_posix()):
            write_pickle(file, pickle_name, config.met_path)
            return


def get_spr_paths_dict():
    """
    Saves ordered spreadsheet paths as a pickled dict
    spr_paths_dict[preprocess_key] = spr_path.

    Returns
    -------
    spr_paths_dict: dict of pathlib.Path with tuple keys
        Ordered spreadsheet paths
    """
    spreadsheets = [file for file in config.spr_path.glob("**/*") if file.is_file()]
    spr_paths_dict = {}
    for key in config.preprocess_key_list:
        for file in spreadsheets:
            if re.search(config.spr_pattern.format(*key), file.as_posix()):
                spr_paths_dict[key] = file
    write_pickle(spr_paths_dict, "spr_paths_dict.pickle", config.met_path)

    # return spr_paths_dict


def main():
    """
    Define mouse IDs, day, trial and camera numbers. Then, get spreadsheet paths.
    """
    # spr_paths_dict = get_spr_paths_dict()
    get_spr_paths_dict()
    spr_paths_dict = read_pickle("spr_paths_dict.pickle", config.met_path)
    xys = {}
    lhs = {}
    for key, path in spr_paths_dict.items():
        kwargs = {"skiprows": config.spr_skiprows}
        xy_kwargs = {**kwargs, "usecols": config.spr_xy_columns}
        lh_kwargs = {**kwargs, "usecols": config.spr_lh_columns}
        try:
            xys[key] = pd.read_csv(path, **xy_kwargs).dropna(how="all").values
            lhs[key] = pd.read_csv(path, **lh_kwargs).dropna(how="all").values
        except:
            xys[key] = pd.read_excel(path, **xy_kwargs).dropna(how="all").values
            lhs[key] = pd.read_excel(path, **lh_kwargs).dropna(how="all").values

    return xys, lhs


##################################### MAIN #####################################

if __name__ == "__main__":
    get_spr_paths_dict()

    xys, lhs = main()
    print(xys)
