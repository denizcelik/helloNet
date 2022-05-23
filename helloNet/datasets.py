import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from helloNet.preprocessing import flatten_data, normalize_image_data
from time import time
from PIL import Image


def create_image_classification_dataset(
    path_dataset_root,
    size_px=128,
    ratio_test_set=0,
    ratio_val_set=0,
    shuffle=True,
    channels=3,
    show_eleminate_log=True,
    sw_easter=True,
):

    t1 = time()
    print("The Processing Log:\n")

    print("Importing dataset...")

    # Get directories in the root file as classes
    classes = os.listdir(path_dataset_root)

    if not path_dataset_root.endswith("/"):
        path_dataset_root = path_dataset_root + "/"

    #  Easter egg switch
    sw_easter = True

    X_set = []
    Y_set = []

    counter_total = 0
    counter_added = 0

    list_broken_files = []

    for class_obj in classes:

        # Modify path for current class
        path_current_class = path_dataset_root + class_obj

        # Get file (image) names in current class folder
        names_sample = os.listdir(path_current_class)

        # ADD: Arithmetic reorder of sample names
        # ADD: File type elemination
        # ADD: Dimensions (channel) elemination

        # Get number of image file samples
        num_sample_class = len(names_sample)

        # Count total set examples
        counter_total += num_sample_class

        for sample_name in names_sample:

            # Modify the path for current image file
            path_current_sample = path_current_class + "/" + sample_name

            # Get size of current sample
            size_of_sample = os.stat(path_current_sample).st_size

            # Eleminate 0 byte files
            if size_of_sample > 0:

                # Load current image with PIL.Image module
                img_sample = Image.open(path_current_sample)
                # print(
                #     "size:",
                #     img_sample.size,
                #     "format:",
                #     img_sample.format,
                #     "Type:",
                #     type(img_sample),
                #     "mode:",
                #     img_sample.mode,
                #     "name:",
                #     sample_name,
                #     "class:",
                #     class_obj,
                # )

                # Resize image to fixed width and height values of the dataset (selected by user)
                img_sample_resized = img_sample.resize((size_px, size_px))

                # Turn PIL image to numpy array
                img_sample_arr = np.array(img_sample_resized)

                # Print broken shapes
                if img_sample_arr.shape != (size_px, size_px, channels):

                    # Print easter egg
                    if sw_easter:
                        print("*green light... tik... tok... red light...*")
                        print("Eleminating broken files.")
                        sw_easter = False

                    # Stack broken file names
                    file_broken = [class_obj, sample_name]
                    list_broken_files.append(file_broken)

                    if show_eleminate_log:
                        #  Print broken file names and its class
                        print(f"File eleminated: {class_obj} - {sample_name}")

                else:
                    # Add current image array to X and Y lists
                    X_set.append(img_sample_arr)
                    Y_set.append(classes.index(class_obj))
                    counter_added += 1

    # Turn lists to array
    X_set = np.array(X_set)
    Y_set = np.array(Y_set)

    # Print class names
    print("\nClasses of The Dataset:", classes)

    # Print input data example size
    print(f"Data Input Size: {counter_total} files")

    # Get number of set examples
    m_set = X_set.shape[0]

    # Sanity Check
    if m_set != counter_added:
        print("ERROR")

    # Get number of set examples
    m_set = X_set.shape[0]
    print(f"Dataset Size: {m_set} images")
    print(f"Detected broken files: {counter_total - m_set}")

    # Order set indices randomly
    if shuffle:
        indexes_random_ord = np.random.permutation(m_set)
        X_set = X_set[indexes_random_ord]
        Y_set = Y_set[indexes_random_ord]
        print("Shuffled randomly.")

    list_root_path = path_dataset_root.split("/")
    # print("list_root_path", list_root_path)

    name_root = list_root_path.pop(-2)
    # print("name_root", name_root)

    path_root_parent = "/".join(list_root_path)
    print("path_root_parent", path_root_parent)

    path_root_parent = "saved_datasets"
    os.makedirs(path_root_parent, exist_ok=True)

    path_dataset_save = path_root_parent + "/dataset-" + name_root
    # print(path_dataset_save)

    os.mkdir(path_dataset_save)
    # print("Dataset directory created.")

    name_dataset_classes = path_dataset_save + "/" + f"classes_{name_root}.h5"
    hf_object_cls = h5py.File(name_dataset_classes, "w")
    hf_object_cls.create_dataset("classes", data=classes)
    hf_object_cls.close()

    partitions = 1
    if ratio_test_set > 0:
        partitions += 1
    if ratio_val_set > 0:
        partitions += 1

    # Check split conditions
    if partitions > 1:

        # Get split ratio for training part
        ratio_split = ratio_test_set + ratio_val_set

        # Check ratio input conditions
        if ratio_split > 0.9:
            raise ValueError(
                f"Invalid value. Please check '{f'{ratio_test_set=}'.split('=')[0]}' and '{f'{ratio_val_set=}'.split('=')[0]}' variables."
            )

        # Get split index for training part
        index_split = int(m_set - m_set * ratio_split)
        print(f"Split index-1: {index_split}")

        # If we split to three part: 'train', 'val','test
        if partitions == 3:

            index_split_test = int(
                (ratio_val_set / (ratio_test_set + ratio_val_set))
                * (m_set - index_split)
                + index_split
            )
            print(f"Split index-2: {index_split_test}")

            X_set_train = X_set[:index_split]
            Y_set_train = Y_set[:index_split]
            Y_set_train = Y_set_train.reshape(1, Y_set_train.shape[0])

            X_set_val = X_set[index_split:index_split_test]
            Y_set_val = Y_set[index_split:index_split_test]
            Y_set_val = Y_set_val.reshape(1, Y_set_val.shape[0])

            X_set_test = X_set[index_split_test:]
            Y_set_test = Y_set[index_split_test:]
            Y_set_test = Y_set_test.reshape(1, Y_set_test.shape[0])

            print(
                f"\nShapes:",
                f"Train set:\t {X_set_train.shape}",
                f"Train labels:\t {Y_set_train.shape}",
                f"Val. set:\t {X_set_val.shape}",
                f"Val. labes:\t {Y_set_val.shape}",
                f"Test set:\t {X_set_test.shape}",
                f"Test labes:\t {Y_set_test.shape}",
                sep="\n",
            )

            name_dataset_train = path_dataset_save + "/" + f"train_{name_root}.h5"
            hf_object = h5py.File(name_dataset_train, "w")
            hf_object.create_dataset("X_train", data=X_set_train)
            hf_object.create_dataset("Y_train", data=Y_set_train)
            hf_object.close()

            name_dataset_test = path_dataset_save + "/" + f"test_{name_root}.h5"
            hf_object_test = h5py.File(name_dataset_test, "w")
            hf_object_test.create_dataset("X_test", data=X_set_test)
            hf_object_test.create_dataset("Y_test", data=Y_set_test)
            hf_object_test.close()

            name_dataset_val = path_dataset_save + "/" + f"val_{name_root}.h5"
            hf_object_val = h5py.File(name_dataset_val, "w")
            hf_object_val.create_dataset("X_val", data=X_set_val)
            hf_object_val.create_dataset("Y_val", data=Y_set_val)
            hf_object_val.close()

        elif partitions == 2:

            X_set_train = X_set[:index_split]
            Y_set_train = Y_set[:index_split]
            Y_set_train = Y_set_train.reshape(1, Y_set_train.shape[0])

            X_set_test = X_set[index_split:]
            Y_set_test = Y_set[index_split:]
            Y_set_test = Y_set_test.reshape(1, Y_set_test.shape[0])

            print(
                f"\nShapes:",
                f"Train set:\t {X_set_train.shape}",
                f"Train labels:\t {Y_set_train.shape}",
                f"Test set:\t {X_set_test.shape}",
                f"Test labes:\t {Y_set_test.shape}",
                sep="\n",
            )

            name_dataset_train = path_dataset_save + "/" + f"train_{name_root}.h5"
            hf_object = h5py.File(name_dataset_train, "w")
            hf_object.create_dataset("X_train", data=X_set_train)
            hf_object.create_dataset("Y_train", data=Y_set_train)
            hf_object.close()

            name_dataset_test = path_dataset_save + "/" + f"test_{name_root}.h5"
            hf_object_test = h5py.File(name_dataset_test, "w")
            hf_object_test.create_dataset("X_test", data=X_set_test)
            hf_object_test.create_dataset("Y_test", data=Y_set_test)
            hf_object_test.close()

    # No split option (returns whole set as one part)
    elif partitions == 1:

        Y_set = Y_set.reshape(1, Y_set.shape[0])

        print(
            f"\nShapes:",
            f"Train set:\t {X_set.shape}",
            f"Train labels:\t {Y_set.shape}",
            sep="\n",
        )

        name_dataset_train = path_dataset_save + "/" + f"train_{name_root}.h5"
        hf_object = h5py.File(name_dataset_train, "w")
        hf_object.create_dataset("X_train", data=X_set)
        hf_object.create_dataset("Y_train", data=Y_set)
        hf_object.close()

    # Print dataset create process information
    print(
        f"\nDataset files created.\nYou can find your dataset files in 'dataset-{name_root}' directory."
    )

    name_broken_files = path_dataset_save + "/" + f"broken_files_{name_root}.txt"
    with open(name_broken_files, "w") as text_file:
        for item in list_broken_files:
            text_file.write(f"class: {item[0]}, file: {item[1]}\n")

    # Print text log create process information
    print(
        f"Broken files log created.\nYou can find the log file in 'dataset-{name_root}' directory."
    )

    t2 = time()
    exec_time = t2 - t1
    print(f"Execution time: {int(exec_time/60)} mins. {int((t2-t1)%60)} secs.")


def load_image_classification_dataset(name_dataset="", path_full=None):

    if path_full is None:
        path_root_dataset = "helloNet/datasets/dataset-" + name_dataset
    else:
        path_root_dataset = path_full

    if not path_root_dataset.endswith("/"):
        path_root_dataset = path_root_dataset + "/"

    list_dataset_elements = os.listdir(path_root_dataset)

    my_dataset = {}

    for elm in list_dataset_elements:

        path_elm_name = path_root_dataset + elm

        if elm.startswith("classes_"):

            print("Imported:", elm)
            element = h5py.File(path_elm_name, "r")
            classes = np.array(element.get("classes"))
            my_dataset["classes"] = classes

        elif elm.startswith("train_"):

            print("Imported:", elm)
            element = h5py.File(path_elm_name, "r")
            X_train = np.array(element.get("X_train"))
            Y_train = np.array(element.get("Y_train"))
            if Y_train.shape != (1, X_train.shape[0]):
                Y_train = Y_train.reshape(1, Y_train.shape[0])
            my_dataset["X_train"] = X_train
            my_dataset["Y_train"] = Y_train

        elif elm.startswith("val_"):

            print("Imported:", elm)
            element = h5py.File(path_elm_name, "r")
            X_val = np.array(element.get("X_val"))
            Y_val = np.array(element.get("Y_val"))
            if Y_val.shape != (1, X_val.shape[0]):
                Y_val = Y_val.reshape(1, Y_val.shape[0])
            my_dataset["X_val"] = X_val
            my_dataset["Y_val"] = Y_val

        elif elm.startswith("test_"):

            print("Imported:", elm)
            element = h5py.File(path_elm_name, "r")
            X_test = np.array(element.get("X_test"))
            Y_test = np.array(element.get("Y_test"))
            if Y_test.shape != (1, X_test.shape[0]):
                Y_test = Y_test.reshape(1, Y_test.shape[0])
            my_dataset["X_test"] = X_test
            my_dataset["Y_test"] = Y_test

    print(f"\nClass names: {classes} and Shape: {classes.shape}")

    for key in my_dataset.keys():

        if key.startswith("X_"):
            key_set = key.split("_")[1]
            key_set_cap = key_set[0].upper() + key_set[1:]
            print(
                f"{key_set_cap} Set - Size: {my_dataset[key].shape[0]}\t- Shapes: {key}:\t{my_dataset[key].shape},\tLabels ({'Y_'+key_set}): {my_dataset['Y_'+key_set].shape}"
            )

    print(f"\nReturned: {', '.join(my_dataset)} in a dictionary")

    return my_dataset


def load_test_sample(path_sample, size_px_train):

    # Lazy read image with PIL.Image module
    img_sample = Image.open(path_sample)

    # Resize image to fixed width and height values of the dataset (selected by user)
    img_sample_resized = img_sample.resize((size_px_train, size_px_train))

    # Turn PIL image to numpy array
    img_sample_resized = np.array(img_sample_resized)

    img_sample_extended = np.array([img_sample_resized])

    # Reshape input to implement into neural net
    # n_features = img_sample_resized.size
    # img_sample_resized = img_sample_resized.reshape((n_features, 1))

    img_sample_flatten = flatten_data(img_sample_extended)

    img_sample_normalized = normalize_image_data(img_sample_flatten)

    # Showing example
    plt.imshow(img_sample)

    return img_sample_normalized


def split_train_test(X_set, Y_set, split_ratio_test, shuffle=False):

    # Get input set size
    m_set = X_set.shape[0]

    #  Check for randomly shuffle the set
    if shuffle:
        indexes_random = np.random.permutation(m_set)
        X_set = X_set[indexes_random]
        Y_set = Y_set[indexes_random]

    # Get split index for train set
    index_train = int(np.floor(m_set - (m_set * split_ratio_test)))

    # Split train set and labels
    X_train = X_set[:index_train]
    Y_train = Y_set[:index_train]

    # Split test set and labels
    X_test = X_set[index_train:]
    Y_test = Y_set[index_train:]

    #  Print final informations
    print(
        "New Shapes",
        f"X_train: {X_train.shape}, Y_train:\t {Y_train.shape}",
        f"X_test:\t {X_test.shape}, Y_test:\t {Y_test.shape}",
        sep="\n",
    )

    # Return splitted sets
    return X_train, Y_train, X_test, Y_test


def split_train_val_test(
    X_set, Y_set, split_ratio_val, split_ratio_test, shuffle=False
):

    # Get input set size
    m_set = X_set.shape[0]

    #  Check for randomly shuffle the set
    if shuffle:
        indexes_random = np.random.permutation(m_set)
        X_set = X_set[indexes_random]
        Y_set = Y_set[indexes_random]

    # Get total ratio of val and test sets
    split_ratio = split_ratio_test + split_ratio_val

    # Get ratio for between val and test sets
    ratio_val_test = split_ratio_val / (split_ratio_val + split_ratio_test)

    #  Turn ratio into split ratio
    split_ratio_val = split_ratio * ratio_val_test

    # Get split index for train set
    index_train = int(np.floor(m_set - (m_set * split_ratio)))

    # Get split index for val set
    index_val = int(np.floor(index_train + (m_set * split_ratio_val)))

    # Split train set and labels
    X_train = X_set[:index_train]
    Y_train = Y_set[:index_train]

    # Split val set and labels
    X_val = X_set[index_train:index_val]
    Y_val = Y_set[index_train:index_val]

    # Split test set and labels
    X_test = X_set[index_val:]
    Y_test = Y_set[index_val:]

    #  Print final informations
    print(
        "New Shapes",
        f"X_train: {X_train.shape}, Y_train:\t {Y_train.shape}",
        f"X_val:\t {X_val.shape}, Y_val:\t {Y_val.shape}",
        f"X_test:\t {X_test.shape}, Y_test:\t {Y_test.shape}",
        sep="\n",
    )

    # Return splitted sets
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
