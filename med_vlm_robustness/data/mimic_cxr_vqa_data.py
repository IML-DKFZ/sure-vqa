from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import re
from tqdm import tqdm


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to the MIMIC-CXR-VQA dataset. Should contain a subfolder mimic-cxr-vqa with the same structure "
             "as described in the README.",
        default=None,
    )

    args = parser.parse_args()
    return args


def get_list_categories(question_series):
    if question_series["semantic_type"] == "verify":
        return ["yes", "no"]
    elif question_series["semantic_type"] == "choose":
        if len(re.findall("\${\w*} or (?:the )?\${\w*}", question_series["template"])) != 1:
            try:
                options = re.findall(", \w* or \w*", question_series["question"])[0].split(", ")[1].split(" or ")
            except IndexError:
                if question_series["answer_orig"][0] in ["ap", "pa"]:
                    options = ["AP", "PA"]
                else:
                    raise
            assert len(set(options).intersection(["AP","PA"])) == 2 or \
                   len(set(options).intersection(["male","female"])) == 2
            return options
        #assert len(re.findall("\${\w*} or (?:the )?\${\w*}", question_series["template"])) == 1
        arguments = re.findall("\${\w*} or (?:the )?\${\w*}", question_series["template"])[0].replace(
            "${", "").replace("}", "")
        arguments = re.split(" or (?:the )?", arguments)
        assert len(arguments) == 2
        options = []
        for argument in arguments:
            kind, number = argument.split("_")
            options.append(question_series["template_arguments"][kind][str(int(number)-1)])
        return [*options, "none", "both"]
    else:
        return None


def add_num_categories(question_series):
    if question_series["list_categories"] is None:
        return None
    return len(question_series["list_categories"])


def add_answer_type(question_series):
    if question_series["semantic_type"] == "verify" or question_series["semantic_type"] == "choose":
        return "CLOSED"
    else:
        return "OPEN"


def modify_answers(question_series):
    if question_series["semantic_type"] == "verify":
        if question_series["answer_orig"][0] == "yes":
            return "yes"
        elif question_series["answer_orig"][0] == "no":
            return "no"
        else:
            raise ValueError
    elif question_series["semantic_type"] == "choose":
        if len(question_series["answer_orig"]) == 1:
            if len(set(question_series["list_categories"]).intersection(["male","female"])) == 2:
                if question_series["answer_orig"][0] == "f":
                    return "female"
                return "male"
            return question_series["answer_orig"][0]
        elif len(question_series["answer_orig"]) == 2:
            return "both"
        elif len(question_series["answer_orig"]) == 0:
            return "none"
        else:
            raise ValueError
    else:
        if len(question_series["answer_orig"]) == 0:
            return "none"
        if question_series["content_type"] == "gender":
            if question_series["answer_orig"][0] == "f":
                return "female"
            elif question_series["answer_orig"][0] == "m":
                return "male"
        return ", ".join(question_series["answer_orig"])


def map_patient_gender(question_series, patient_metadata):
    subject_id = question_series["subject_id"]
    patient_metadata = patient_metadata[patient_metadata["subject_id"] == subject_id]
    if patient_metadata['gender'].nunique() > 1:
        print(f"Genders of patient {subject_id} are not unique.")
        print(patient_metadata["gender"])
        gender = None
    elif patient_metadata['gender'].nunique() == 0:
        gender = None
    else:
        gender = patient_metadata.iloc[0]["gender"]
    return gender


def map_anchor_age(question_series, patient_metadata):
    subject_id = question_series["subject_id"]
    patient_metadata = patient_metadata[patient_metadata["subject_id"] == subject_id]
    if patient_metadata['anchor_age'].nunique() > 1:
        print(f"Anchor age of patient {subject_id} is not unique.")
        print(patient_metadata['anchor_age'])
        anchor_age = None
    elif patient_metadata['anchor_age'].nunique() == 0:
        anchor_age = None
    else:
        anchor_age = patient_metadata.iloc[0]["anchor_age"]
    return anchor_age


def map_anchor_year_group(question_series, patient_metadata):
    subject_id = question_series["subject_id"]
    patient_metadata = patient_metadata[patient_metadata["subject_id"] == subject_id]
    if patient_metadata['anchor_year_group'].nunique() > 1:
        print(f"Anchor year group of patient {subject_id} is not unique.")
        print(patient_metadata['anchor_year_group'])
        anchor_year_group = None
    elif patient_metadata['anchor_year_group'].nunique() == 0:
        anchor_year_group = None
    else:
        anchor_year_group = patient_metadata.iloc[0]["anchor_year_group"]
    return anchor_year_group


def map_ethnicity(question_series, admissions):
    white = ['WHITE', 'WHITE - RUSSIAN', 'PORTUGUESE', 'WHITE - OTHER EUROPEAN', 'WHITE - BRAZILIAN',
             'WHITE - EASTERN EUROPEAN']
    unknown_other = ['OTHER', 'UNABLE TO OBTAIN', 'UNKNOWN', 'PATIENT DECLINED TO ANSWER',
                     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKA NATIVE',
                     'MULTIPLE RACE/ETHNICITY', 'SOUTH AMERICAN']
    black_african = ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/AFRICAN', 'BLACK/CARIBBEAN ISLAND', ]
    asian = ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - SOUTH EAST ASIAN', 'ASIAN - KOREAN', 'ASIAN - ASIAN INDIAN']
    hispanic_latino = ['HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - SALVADORAN', 'HISPANIC/LATINO - PUERTO RICAN',
                       'HISPANIC/LATINO - GUATEMALAN', 'HISPANIC OR LATINO', 'HISPANIC/LATINO - MEXICAN',
                       'HISPANIC/LATINO - CUBAN', 'HISPANIC/LATINO - HONDURAN', 'HISPANIC/LATINO - COLUMBIAN',
                       'HISPANIC/LATINO - CENTRAL AMERICAN']

    ethnicity_categories = {
        "WHITE": white,
        "UNKNOWN/OTHER": unknown_other,
        "BLACK": black_african,
        "ASIAN": asian,
        "HISPANIC/LATINO": hispanic_latino
    }
    subject_id = question_series["subject_id"]
    admissions = admissions[admissions["subject_id"] == subject_id]
    ethnicities = list(admissions["race"])
    matched_categories = []
    for ethnicity_key, ethnicity_values in ethnicity_categories.items():
        if len(set(ethnicities).intersection(ethnicity_values)) > 0:
            matched_categories.append(ethnicity_key)
    if len(matched_categories) > 1 and "UNKNOWN/OTHER" in matched_categories:
        matched_categories.remove("UNKNOWN/OTHER")
    if len(matched_categories) == 0:
        matched_categories.append("UNKNOWN/OTHER")
    if len(matched_categories) != 1:
        # print("Multiple categories: ", matched_categories)
        return "UNKNOWN/OTHER"
    return matched_categories[0]


def prepare_data(args):
    dataset_path = Path(args.path)
    questions_path = dataset_path / "mimic-cxr-vqa" / "mimiccxrvqa" / "dataset"
    train_questions_file = questions_path / "train.json"
    val_questions_file = questions_path / "valid.json"
    test_questions_file = questions_path / "test.json"
    patient_metadata_file = dataset_path / "mimic-cxr-vqa" / "physionet.org" / "files" / "mimiciv" / "2.2" / "hosp" / "patients.csv"
    admissions_file = dataset_path / "mimic-cxr-vqa" / "physionet.org" / "files" / "mimiciv" / "2.2" / "hosp" / "admissions.csv"
    train_questions = pd.read_json(train_questions_file)
    val_questions = pd.read_json(val_questions_file)
    test_questions = pd.read_json(test_questions_file)
    patient_metadata = pd.read_csv(patient_metadata_file)
    admissions = pd.read_csv(admissions_file)
    all_questions = {"train": train_questions, "validate": val_questions, "test": test_questions}
    for question_df in tqdm(all_questions.values()):
        question_df.rename(columns={"answer": "answer_orig", "idx": "qid"}, inplace=True)
        question_df["list_categories"] = question_df.apply(get_list_categories, axis=1)
        question_df["num_categories"] = question_df.apply(add_num_categories, axis=1)
        question_df["answer_type"] = question_df.apply(add_answer_type, axis=1)
        question_df["answer"] = question_df.apply(modify_answers, axis=1)
        question_df["gender"] = question_df.apply(map_patient_gender, args=(patient_metadata,), axis=1)
        question_df["age"] = question_df.apply(map_anchor_age, args=(patient_metadata,), axis=1)
        question_df["year"] = question_df.apply(map_anchor_year_group, args=(patient_metadata,), axis=1)
        question_df["ethnicity"] = question_df.apply(map_ethnicity, args=(admissions,), axis=1)
    for key, value in all_questions.items():
        value.to_json(dataset_path / f"{key}.json", orient="records", lines=False, indent=4)
    print()

if __name__== "__main__":
    args = main_cli()
    prepare_data(args)