import torch.utils.data
import matplotlib.pyplot as plt
import os
import csv

from part_a.utils import *

student_data_path = "data/student_meta.csv"


def load_data(base_path="data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """

    train_path = os.path.join(base_path, "train_sparse.npz")
    train_matrix = load_train_sparse(train_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    # Data columns: user_id,gender,data_of_birth,premium_pupil
    data = {
        "user_id": [],
        "gender": [],
        "year_of_birth": [],
        "premium_pupil": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
                data["year_of_birth"].append(row[2][:4])
                data["premium_pupil"].append(int(row[3]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def group_user_id_by_age(datapath=student_data_path) -> dict:
    """
    Return dictionary of user_id grouped by birth year.
    The groups are <2004, 2004-2005, >2005
    Keys: string of birth year
    Values: list of user_id
    """

    student_data = _load_csv(datapath)
    user_id = student_data["user_id"]
    birth_year = student_data["year_of_birth"]

    user_id_by_age = {
        "<2004": [],
        "2004-2005": [],
        ">2005": []
    }

    for i in range(len(user_id)):
        if birth_year[i] != "":
            year = int(birth_year[i])
            if year < 2004:
                user_id_by_age["<2004"].append(user_id[i])
            elif year < 2006:
                user_id_by_age["2004-2005"].append(user_id[i])
            else:
                user_id_by_age[">2005"].append(user_id[i])

    return user_id_by_age


def group_user_id_by_gender(datapath=student_data_path):
    """
    Return dictionary of user_id grouped by gender.
    The groups are 0, 1, 2
    Keys: int of gender
    Values: list of user_id
    """
    student_data = _load_csv(datapath)
    user_id = student_data["user_id"]
    genders = student_data["gender"]

    user_id_by_gender = {
        0: [],
        1: [],
        2: []
    }

    for i in range(len(user_id)):

        if genders[i] != "":
            gender = int(genders[i])
            if gender == 0:
                user_id_by_gender[0].append(user_id[i])
            elif gender == 1:
                user_id_by_gender[1].append(user_id[i])
            else:
                user_id_by_gender[2].append(user_id[i])

    return user_id_by_gender


def generate_age_data(path, age):
    age_groups = group_user_id_by_age()
    if age == 0:
        user_ids = age_groups["<2004"]
    elif age == 1:
        user_ids = age_groups["2004-2005"]
    else:
        user_ids = age_groups[">2005"]

    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data. Only add data if user_id is in user_ids
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                if int(row[1]) in user_ids:
                    data["question_id"].append(int(row[0]))
                    data["user_id"].append(int(row[1]))
                    data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def generate_gender_data(path, gender):
    gender_groups = group_user_id_by_gender()
    if gender == 0:
        user_ids = gender_groups[0]
    elif gender == 1:
        user_ids = gender_groups[1]
    else:
        user_ids = gender_groups[2]

    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data. Only add data if user_id is in user_ids
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                if int(row[1]) in user_ids:
                    data["question_id"].append(int(row[0]))
                    data["user_id"].append(int(row[1]))
                    data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def plot_age_groups(old_train_losses, old_train_accuracies, old_valid_losses, old_valid_accuracies,
                    med_train_losses, med_train_accuracies, med_valid_losses, med_valid_accuracies,
                    young_train_losses, young_train_accuracies, young_valid_losses, young_valid_accuracies):
    """ So sorry in advance. Felt like the easiest way to plot everything."""
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].set_title('Training Loss over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].plot(old_train_losses, color='xkcd:sky blue', label='Age <2004')
    axes[0, 0].plot(med_train_losses, color='xkcd:royal blue', label='Age 2004-2005')
    axes[0, 0].plot(young_train_losses, color='xkcd:dark blue', label='Age >2005')

    axes[0, 1].set_title('Validation Loss over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].plot(old_valid_losses, color='xkcd:sky blue')
    axes[0, 1].plot(med_valid_losses, color='xkcd:royal blue')
    axes[0, 1].plot(young_valid_losses, color='xkcd:dark blue')

    axes[1, 0].set_title('Training Accuracy over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].plot(old_train_accuracies, color='xkcd:sky blue')
    axes[1, 0].plot(med_train_accuracies, color='xkcd:royal blue')
    axes[1, 0].plot(young_train_accuracies, color='xkcd:dark blue')

    axes[1, 1].set_title('Validation Accuracy over Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].plot(old_valid_accuracies, color='xkcd:sky blue')
    axes[1, 1].plot(med_valid_accuracies, color='xkcd:royal blue')
    axes[1, 1].plot(young_valid_accuracies, color='xkcd:dark blue')

    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

