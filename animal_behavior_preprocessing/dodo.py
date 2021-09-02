"""Dodo go brrr.

BRRRR.
"""
import config
from spreadsheet_handle import get_spreadsheet_path, get_median_filter_perspective
from kalman_filter import get_kalman_filter
from joint_angles import get_joint_angles_statistics
from wavelet_spectra import get_wavelet_spectra


def task_get_spreadsheet_path():
    """Spreadsheet handling.

    Find, open and read motion tracking spreadsheets (CSV or Excel files).
    Save spreadsheet paths.
    """
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
    """Median filter and perspective transform matrix.

    Apply median transform to X-Y coords and likelihoods of body-part markers.
    Compute perspective transform matrices of front and back cameras.
    Save results.
    """
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
    """Kalman filter of combined front and back camera's motion tracking.

    Combine front and back body-part markers, in the same perspective.
    Apply parallelized Ensemble Kalman filter to combined coordinates.
    Save results.
    """
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


def task_get_joint_angles_statistics():
    """Computes joint angles and their statistics.

    Computes new X-Y coords, with and additional mid-bottom-rotarod marker.
    Computes joint angles defined by some select thruples of markers.
    Computes number of frames, means and variances for each single angle."""
    for key in config.key_list:
        name = config.subject_name.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_path = config.kal_path / f"kal_xy_{pickle_end}"
        target_pickle_paths = {
            "ang": config.ang_path / f"ang_{pickle_end}",
            "stat": config.ang_path / f"stat_{pickle_end}",
        }
        yield {
            "name": name,
            "file_dep": [dep_pickle_path],
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_joint_angles_statistics,
                    [dep_pickle_path, target_pickle_paths],
                )
            ],
        }


def task_get_wavelet_spectra():
    """Computes wavelet spectra of whitened joint angles."""
    for key in config.key_list:
        name = config.subject_name.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths = {
            "ang": config.ang_path / f"ang_{pickle_end}",
            "stat": config.ang_path / f"stat_{pickle_end}",
        }
        target_pickle_path = config.wav_path / f"wav_{pickle_end}"
        yield {
            "name": name,
            "file_dep": [dep_pickle_paths],
            "targets": list(target_pickle_path.values()),
            "actions": [
                (
                    get_wavelet_spectra,
                    [dep_pickle_paths, target_pickle_path],
                )
            ],
        }


# ##################################### MAIN #####################################

# if __name__ == "__main__":
#     import doit

#     doit.run(globals())
