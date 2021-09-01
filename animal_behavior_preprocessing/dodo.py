"""Dodo go brrr.

Create folders, write and read pickles, get spreadsheet paths and load data.
"""
import config
from spreadsheet_handle import get_spreadsheet_path, get_median_filter_perspective
from kalman_filter import get_kalman_filter


def task_get_spreadsheet_path():
    """Get spreadsheet path go brrr."""
    spreadsheets = [file for file in config.spr_path.glob("**/*") if file.is_file()]
    for key in config.preprocess_key_list:
        name = config.preprocess_subject_name.format(*key)
        pickle_end = name + ".pickle"
        target_pickle_path = config.met_path / f"spr_path_{pickle_end}"
        yield {
            "name": name,
            "targets": [target_pickle_path],
            "actions": [
                (
                    get_spreadsheet_path,
                    [spreadsheets, key, target_pickle_path],
                )
            ],
        }


def task_get_median_filter_perspective():
    """Get median filter and perspective matrix go brrr."""
    for key in config.preprocess_key_list:
        name = config.preprocess_subject_name.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_path = config.met_path / f"spr_path_{pickle_end}"
        target_pickle_paths = {
            "med_xy": config.med_path / f"med_xy_{pickle_end}",
            "med_lh": config.med_path / f"med_lh_{pickle_end}",
            "perspective": config.med_path / f"perspective_{pickle_end}",
        }
        yield {
            "name": name,
            "file_dep": [dep_pickle_path],
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_median_filter_perspective,
                    [dep_pickle_path, target_pickle_paths],
                )
            ],
        }


def task_get_kalman_filter():
    """Get kalman filter go brrr."""
    for key in config.key_list:
        name = config.subject_name.format(*key)
        pickle_end = name + ".pickle"
        front_name = config.preprocess_subject_name.format(*key, 1)
        front_pickle_end = front_name + ".pickle"
        back_name = config.preprocess_subject_name.format(*key, 2)
        back_pickle_end = back_name + ".pickle"
        dep_pickle_paths = {
            "front_med_xy": config.med_path / f"med_xy_{front_pickle_end}",
            "front_med_lh": config.med_path / f"med_lh_{front_pickle_end}",
            "front_perspective": config.med_path / f"perspective_{front_pickle_end}",
            "back_med_xy": config.med_path / f"med_xy_{back_pickle_end}",
            "back_med_lh": config.med_path / f"med_lh_{back_pickle_end}",
            "back_perspective": config.med_path / f"perspective_{back_pickle_end}",
        }
        target_pickle_path = config.kal_path / f"kal_xy_{pickle_end}"
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": [target_pickle_path],
            "actions": [
                (
                    get_kalman_filter,
                    [dep_pickle_paths, target_pickle_path],
                )
            ],
        }


# ##################################### MAIN #####################################

# if __name__ == "__main__":
#     import doit

#     doit.run(globals())
