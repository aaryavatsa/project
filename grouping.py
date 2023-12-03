"""
Functions to load student_meta.csv and group them in age groups and gender groups.
"""

import os
import csv


student_data_path = "data/student_meta.csv"

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
