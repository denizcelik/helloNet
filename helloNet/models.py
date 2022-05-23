import pickle
from copy import deepcopy
from os import makedirs
from datetime import datetime


def save_model(
    parameters_model,
    layers_hidden,
    num_output_unit,
    classes_model,
    name_model,
    time_suffix=False,
):
    parameters_my_model_copy = deepcopy(parameters_model)
    my_model = {}

    my_model["parameters"] = parameters_my_model_copy
    my_model["classes"] = classes_model

    layers_my_model = deepcopy(layers_hidden)
    layers_my_model.append(num_output_unit)

    my_model["layers"] = layers_my_model
    my_model["info"] = ""

    content_model = "model content (keys): " + ", ".join(my_model) + "\n"
    content_parameters = (
        f"key content - 'parameters'\t- type: '{type(my_model['parameters']).__name__}',\t\tkeys: "
        + ", ".join(my_model["parameters"])
        + "\n"
    )
    content_classes = (
        f"key content - 'classes'\t\t- type: '{type(my_model['classes']).__name__}',\telements: "
        + ", ".join(str(elmnt) for elmnt in my_model["classes"])
        + "\n"
    )
    content_layers = (
        f"key content - 'layers'\t\t- type: '{type(my_model['layers']).__name__}',\t\telements: "
        + ", ".join(str(elmnt) for elmnt in my_model["layers"])
        + "\n"
    )
    content_info = (
        f"key content - 'info'\t\t- type: '{type(my_model['info']).__name__}',\t\t*information text*"
        + "\n"
    )
    content_notif = (
        "\n*The value of 'info' key returns model informations (above) as string.\n"
    )
    content_all = (
        content_model
        + content_parameters
        + content_classes
        + content_layers
        + content_info
    )
    print(content_all, content_notif)

    my_model["info"] = content_all

    dir_models = "saved_models"

    makedirs(dir_models, exist_ok=True)

    if not name_model.endswith(".pickle"):
        name_model = name_model + ".pickle"

    dir_saved_model = dir_models + "/" + name_model

    if time_suffix:
        date = datetime.now().strftime("_%m-%d_%H-%M-%S")
        dir_saved_model = dir_saved_model[:-7] + date + dir_saved_model[-7:]

    with open(dir_saved_model, "wb") as file_handle:
        pickle.dump(my_model, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Your model saved with name '{dir_saved_model}'")


def load_model(name_model_file, print_info=False, path_model_full=None):

    dir_models = "saved_models"

    # Overrides model name if path_model_full is given
    if path_model_full is None:
        name_model_file = dir_models + "/" + name_model_file

    if not name_model_file.endswith(".pickle"):
        name_model_file = name_model_file + ".pickle"

    with open(name_model_file, "rb") as file_hande:
        model = pickle.load(file_hande)

    print(f"The model '{name_model_file}' imported.")

    if print_info:
        print(model["info"])

    return model
