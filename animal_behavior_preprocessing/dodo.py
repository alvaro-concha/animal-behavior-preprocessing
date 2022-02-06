"""Animal Behavior Preprocessing.

Tasks
-----
    Spreadsheet handling.
    Median filter and perspective transform matrix.
    Kalman filter of combined front and back camera's motion tracking.
    Computes joint angles.
    Fits standard scaler to joint angles.
    Computes Morlet wavelet spectra of whitened joint angles.
    Fits incremental PCA, with whitening, to wavelet spectra features.
"""
from spreadsheet_handle import get_spreadsheet_path, get_median_filter_perspective
from kalman_filter import get_kalman_filter
from quantile_filter_smoother import get_quantile_filter_smoother
from metrics import (
    get_latency_to_fall,
    get_step_metrics,
    get_steps_poses_features_subsample_indices,
    get_standard_scaler_fit_steps_poses_features,
    get_pca_fit_steps_poses_features,
)
from joint_angles import get_joint_angles, get_standard_scaler_fit_joint_angles
from wavelet_spectra import get_wavelet_spectra, get_pca_fit_wavelet_spectra
import config_dodo


def task_get_spreadsheet_path():
    """Spreadsheet handling.

    Find, open and read motion tracking spreadsheets (CSV or Excel files).
    Save spreadsheet paths.
    """
    (config_dodo.MET_PATH / "Paths").mkdir(parents=True, exist_ok=True)
    spreadsheets = [
        file for file in config_dodo.SPR_PATH.glob("**/*") if file.is_file()
    ]
    for key in config_dodo.PREPROCESS_KEY_LIST:
        name = config_dodo.PREPROCESS_SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        target_pickle_path = (
            config_dodo.MET_PATH / f"Paths/spreadsheet_path_{pickle_end}"
        )
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
    (config_dodo.QUA_PATH / "Positions/Corners").mkdir(parents=True, exist_ok=True)
    for key in config_dodo.PREPROCESS_KEY_LIST:
        name = config_dodo.PREPROCESS_SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_path = config_dodo.MET_PATH / f"spreadsheet_path_{pickle_end}"
        target_pickle_paths = {
            "med_xy": config_dodo.MED_PATH / f"med_xy_{pickle_end}",
            "med_lh": config_dodo.MED_PATH / f"med_lh_{pickle_end}",
            "perspective": config_dodo.MED_PATH / f"perspective_{pickle_end}",
            "fig_corners": config_dodo.QUA_PATH
            / f"Positions/Corners/corners_{name}.png",
        }
        yield {
            "name": name,
            "file_dep": [dep_pickle_path],
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_median_filter_perspective,
                    [name, key, dep_pickle_path, target_pickle_paths],
                )
            ],
        }


def task_get_kalman_filter():
    """Kalman filter of combined front and back camera's motion tracking.

    Combine front and back body-part markers, in the same perspective.
    Apply parallelized Ensemble Kalman filter to combined coordinates.
    Save results.
    """
    (config_dodo.QUA_PATH / "Positions/Kalman").mkdir(parents=True, exist_ok=True)
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        front_name = config_dodo.PREPROCESS_SUBJECT_NAME.format(*key, 1)
        front_pickle_end = front_name + ".pickle"
        back_name = config_dodo.PREPROCESS_SUBJECT_NAME.format(*key, 2)
        back_pickle_end = back_name + ".pickle"
        dep_pickle_paths = {
            "front_med_xy": config_dodo.MED_PATH / f"med_xy_{front_pickle_end}",
            "front_med_lh": config_dodo.MED_PATH / f"med_lh_{front_pickle_end}",
            "front_perspective": config_dodo.MED_PATH
            / f"perspective_{front_pickle_end}",
            "back_med_xy": config_dodo.MED_PATH / f"med_xy_{back_pickle_end}",
            "back_med_lh": config_dodo.MED_PATH / f"med_lh_{back_pickle_end}",
            "back_perspective": config_dodo.MED_PATH / f"perspective_{back_pickle_end}",
        }
        target_pickle_paths = {
            "kal_xy": config_dodo.KAL_PATH / f"kal_xy_{pickle_end}",
            "fig_kal_x": config_dodo.QUA_PATH / f"Positions/Kalman/kal_x_{name}.png",
            "fig_kal_y": config_dodo.QUA_PATH / f"Positions/Kalman/kal_y_{name}.png",
        }
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_kalman_filter,
                    [name, dep_pickle_paths, target_pickle_paths],
                )
            ],
        }


def task_get_quantile_filter_smoother():
    """Quantile filter and smoother of Kalman filter outputs.

    Apply parallelized quantile filter and smoother to Kalman filter outputs.
    Add extra columns with smoothed coordinates of the nose and
    the center of mass of the mouse to quantile filter outputs.
    Save results.
    """
    (config_dodo.QUA_PATH / "Positions/Quantile").mkdir(parents=True, exist_ok=True)
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_path = config_dodo.KAL_PATH / f"kal_xy_{pickle_end}"
        target_pickle_paths = {
            "qnt_xy": config_dodo.QNT_PATH / f"qnt_xy_{pickle_end}",
            "fig_qnt_x": config_dodo.QUA_PATH / f"Positions/Quantile/qnt_x_{name}.png",
            "fig_qnt_y": config_dodo.QUA_PATH / f"Positions/Quantile/qnt_y_{name}.png",
        }
        yield {
            "name": name,
            "file_dep": [dep_pickle_path],
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_quantile_filter_smoother,
                    [name, dep_pickle_path, target_pickle_paths],
                )
            ],
        }


def task_get_latency_to_fall():
    """Estimate falling time from volatility of center of mass markers."""
    (config_dodo.MTR_PATH / "Latency").mkdir(parents=True, exist_ok=True)
    (config_dodo.QUA_PATH / "Metrics/Latency").mkdir(parents=True, exist_ok=True)
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_path = config_dodo.QNT_PATH / f"qnt_xy_{pickle_end}"
        target_pickle_paths = {
            "latency": config_dodo.MTR_PATH / f"Latency/latency_{pickle_end}",
            "fig_latency": config_dodo.QUA_PATH / f"Metrics/Latency/latency_{name}.png",
        }
        yield {
            "name": name,
            "file_dep": [dep_pickle_path],
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_latency_to_fall,
                    [name, dep_pickle_path, target_pickle_paths],
                )
            ],
        }


def task_get_step_metrics():
    """Detect steps and computes related metrics. Save results."""
    (config_dodo.MTR_PATH / "Step").mkdir(parents=True, exist_ok=True)
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths = {
            "xys": config_dodo.QNT_PATH / f"qnt_xy_{pickle_end}",
            "latency": config_dodo.MTR_PATH / f"Latency/latency_{pickle_end}",
        }
        target_pickle_paths = {
            "events": config_dodo.MTR_PATH / f"Step/step_events_{pickle_end}",
            "statistics": config_dodo.MTR_PATH / f"Step/step_statistics_{pickle_end}",
        }
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_step_metrics,
                    [dep_pickle_paths, target_pickle_paths],
                )
            ],
        }


def task_get_steps_poses_features_subsample_indices():
    """Save steps and poses features and subsample indices."""
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths = {
            "latency": config_dodo.MTR_PATH / f"Latency/latency_{pickle_end}",
            "statistics": config_dodo.MTR_PATH / f"Step/step_statistics_{pickle_end}",
            "events": config_dodo.MTR_PATH / f"Step/step_events_{pickle_end}",
            "xys": config_dodo.QNT_PATH / f"qnt_xy_{pickle_end}",
        }
        target_pickle_paths = {
            "stp": config_dodo.STP_PATH / f"stp_{pickle_end}",
            "idx": config_dodo.IDX_PATH / f"idx_{pickle_end}",
        }
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": list(target_pickle_paths.values()),
            "actions": [
                (
                    get_steps_poses_features_subsample_indices,
                    [dep_pickle_paths, target_pickle_paths],
                )
            ],
        }


def task_get_standard_scaler_fit_steps_poses_features():
    """Fit standard scaler to steps and poses features."""
    dep_pickle_paths = {"stp": []}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["stp"].append(config_dodo.STP_PATH / f"stp_{pickle_end}")
    target_pickle_path = config_dodo.STP_PATH / "standard_scaler_fit_stp.pickle"
    return {
        "file_dep": dep_pickle_paths["stp"],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_standard_scaler_fit_steps_poses_features,
                [dep_pickle_paths, target_pickle_path],
            )
        ],
    }


def task_get_pca_fit_steps_poses_features():
    """Fit incremental PCA to scaled steps and poses features."""
    dep_pickle_paths = {
        "scaler": config_dodo.STP_PATH / "standard_scaler_fit_stp.pickle",
        "stp": [],
    }
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["stp"].append(config_dodo.STP_PATH / f"stp_{pickle_end}")
    target_pickle_path = config_dodo.STP_PATH / "pca_fit_stp.pickle"
    return {
        "file_dep": [dep_pickle_paths["scaler"]] + dep_pickle_paths["stp"],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_pca_fit_steps_poses_features,
                [dep_pickle_paths, target_pickle_path],
            )
        ],
    }


def task_get_joint_angles():
    """Compute joint angles, up to latency to fall time.

    Compute joint angles defined by some select thruples of markers.
    Trimmed mean smoother is applied to the angles.
    Save results.
    """
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths = {
            "xys": config_dodo.QNT_PATH / f"qnt_xy_{pickle_end}",
            "latency": config_dodo.MTR_PATH / f"Latency/latency_{pickle_end}",
        }
        target_pickle_path = config_dodo.ANG_PATH / f"ang_{pickle_end}"
        yield {
            "name": name,
            "file_dep": list(dep_pickle_paths.values()),
            "targets": [target_pickle_path],
            "actions": [
                (
                    get_joint_angles,
                    [dep_pickle_paths, target_pickle_path],
                )
            ],
        }


def task_get_standard_scaler_fit_joint_angles():
    """Fits standard scaler to joint angles."""
    dep_pickle_paths = {"ang": []}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["ang"].append(config_dodo.ANG_PATH / f"ang_{pickle_end}")
    target_pickle_path = config_dodo.ANG_PATH / "standard_scaler_fit_ang.pickle"
    return {
        "file_dep": dep_pickle_paths["ang"],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_standard_scaler_fit_joint_angles,
                [dep_pickle_paths, target_pickle_path],
            )
        ],
    }


def task_get_wavelet_spectra():
    """Computes Morlet wavelet spectra of whitened joint angles.

    Scales joint angles using global mean and variance.
    Computes parallelized Morlet wavelet transform.
    Saves results.
    """
    path_scaler_ang = config_dodo.ANG_PATH / "standard_scaler_fit_ang.pickle"
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        path_ang = config_dodo.ANG_PATH / f"ang_{pickle_end}"
        target_pickle_path = config_dodo.WAV_PATH / f"wav_{pickle_end}"
        yield {
            "name": name,
            "file_dep": [path_scaler_ang, path_ang],
            "targets": [target_pickle_path],
            "actions": [
                (
                    get_wavelet_spectra,
                    [path_scaler_ang, path_ang, target_pickle_path],
                )
            ],
            "verbosity": 2,
        }


def task_get_pca_fit_wavelet_spectra():
    """Fit incremental PCA to wavelet spectra features."""
    dep_pickle_paths = {"wav": []}
    for key in config_dodo.KEY_LIST:
        name = config_dodo.SUBJECT_NAME.format(*key)
        pickle_end = name + ".pickle"
        dep_pickle_paths["wav"].append(config_dodo.WAV_PATH / f"wav_{pickle_end}")
    target_pickle_path = config_dodo.WAV_PATH / "pca_fit_wav.pickle"
    return {
        "file_dep": dep_pickle_paths["wav"],
        "targets": [target_pickle_path],
        "actions": [
            (
                get_pca_fit_wavelet_spectra,
                [dep_pickle_paths, target_pickle_path],
            )
        ],
    }


def main():
    """Run pipeline as a Python script."""
    import doit

    doit.run(globals())


if __name__ == "__main__":
    main()
