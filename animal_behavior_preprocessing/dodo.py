import config
import data_handling

print("preprocess keys")
for key in config.preprocess_key_list:
    print(config.preprocess_subject_name.format(*key))

print("keys")
for key in config.key_list:
    print(config.subject_name.format(*key))


def task_get_spr_paths_dict():
    spreadsheets = [file for file in config.spr_path.glob("**/*") if file.is_file()]
    for key in config.preprocess_key_list:
        pickle_name = (
            "spr_path" + config.preprocess_subject_name.format(*key) + ".pickle"
        )
        yield {
            "name": config.preprocess_subject_name.format(*key),
            "targets": [config.met_path / pickle_name],
            "actions": [(data_handling.get_spr_path, [spreadsheets, key, pickle_name])],
        }


# def task_data_handling():
#     return {
#         "targets": [config.met_path / "spr_paths_dict.pickle"],
#         "actions": [data_handling.get_spr_paths_dict],
#     }


# def task_read_data():
#     return {
#         "targets": [config.met_path / "spr_paths_dict.pickle"],
#         "actions": [data_handling.get_spr_paths_dict],
#     }


# ##################################### MAIN #####################################

# if __name__ == "__main__":
#     import doit

#     doit.run(globals())
