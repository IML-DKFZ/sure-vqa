import ast
import os
from argparse import ArgumentParser
from collections import Counter
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd

def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to the LIDC dataset. Should contain the metadata.csv and annotation_data.csv files. "
             "If None, reads environment variable DATASET_ROOT_DIR/LIDC.",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducibility. Default is 123",
        default=123
    )
    args = parser.parse_args()
    return args

def get_majority_rating(ratings: List[Optional[int]]):
    if ratings.count(None) > 2:
        return None
    ratings = [r for r in ratings if r is not None]
    element_counts = Counter(ratings)
    max_elem = max(set(ratings), key=ratings.count)
    if list(element_counts.values()).count(element_counts[max_elem]) > 1:
        return None
    else:
        return max_elem


def subtlety_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "Extremely Subtle",
        2: "Moderately Subtle",
        3: "Fairly Subtle",
        4: "Moderately Obvious",
        5: "Obvious"
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "How easy is the nodule in the scan to be detected?",
            "answer": f"The nodule in the scan is {category_dict[majority_rating].lower()} to detect",
            "content_type": "subtlety",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the nodule in this scan {category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "subtlety",
            "answer_type": "CLOSED",
        }


def internal_structure_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "Soft Tissue",
        2: "Fluid",
        3: "Fat",
        4: "Air",
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "What is the nodule in this scan internally composed of?",
            "answer": f"The nodule is composed of {category_dict[majority_rating].lower()}",
            "content_type": "internal Structure",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the nodule in this scan composed of {category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "internal Structure",
            "answer_type": "CLOSED",
        }


def calcification_question(ratings: List[int], yes_no: bool):
    # category_dict = {
    #     # 1: "Popcorn",
    #     # 2: "Laminated",
    #     3: "Solid",
    #     # 4: "Non-central",
    #     # 5: "Central",
    #     6: "Absent"
    # }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if yes_no:
        presence_question = random.choices([True, False])[0]
        if presence_question:
            return {
                "question": f"Does the nodule in this scan show calcification?",
                "answer": "Yes" if majority_rating != 6 else "No",
                "content_type": "calcification",
                "answer_type": "CLOSED",
            }
    # if not yes_no:
    #     return {
    #         "question": "If the nodule in this scan shows calcification, what is the pattern of calcification?",
    #         "answer": f"The nodule shows a {category_dict[majority_rating].lower()} calcification pattern."
    #                     if majority_rating != 6 else "There is no calcification present in the nodule.",
    #         "content_type": "calcification",
    #         "answer_type": "OPEN",
    #     }
    # else:
    #     presence_question = random.choices([True, False])[0]
    #     if presence_question:
    #         return {
    #             "question": f"Does the nodule in this scan show calcification?",
    #             "answer": "Yes" if majority_rating != 6 else "No",
    #             "content_type": "calcification",
    #             "answer_type": "CLOSED",
    #         }
        # else:
        #     category_ints = list(category_dict.keys())[:-1]
        #     if majority_rating != 6:
        #         probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        #     else:
        #         probabilities = None
        #     random_element = random.choices(category_ints, weights=probabilities)[0]
        #     return {
        #         "question": f"Does the nodule in this scan show a {category_dict[random_element].lower()} calcification pattern?",
        #         "answer": "Yes" if random_element == majority_rating else "No",
        #         "content_type": "calcification",
        #         "answer_type": "CLOSED",
        #     }


def margin_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "Poorly Defined",
        2: "Near Poorly Defined",
        # Changed from original category "Medium Margin"
        3: "Medium Defined",
        4: "Near Sharp",
        5: "Sharp"
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "How well-defined is the margin of the nodule shown in this scan?",
            "answer": f"The nodule in this image has a {category_dict[majority_rating].lower()} margin",
            "content_type": "margin",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the margin of the nodule in this scan {category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "margin",
            "answer_type": "CLOSED",
        }


def lobulation_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "No Lobulation",
        2: "Nearly No Lobulation",
        3: "Medium Lobulation",
        4: "Near Marked Lobulation",
        5: "Marked Lobulation"
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "What degree of lobulation does the nodule in this scan show?",
            "answer": f"The nodule in this scan shows {category_dict[majority_rating].lower()}",
            "content_type": "lobulation",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Does the nodule in this scan show {category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "lobulation",
            "answer_type": "CLOSED",
        }


def spiculation_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "No Spiculation",
        2: "Nearly No Spiculation",
        3: "Medium Spiculation",
        4: "Near Marked Spiculation",
        5: "Marked Spiculation"
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "What extent of spiculation does the nodule in this scan show?",
            "answer": f"The nodule in this scan shows {category_dict[majority_rating].lower()}",
            "content_type": "spiculation",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Does the nodule in this scan show {category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "spiculation",
            "answer_type": "CLOSED",
        }


def texture_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "Non-Solid/GGO",
        2: "Non-Solid/Mixed",
        3: "Part Solid/Mixed",
        4: "Solid/Mixed",
        5: "Solid"
    }
    answers = {
        1: "The nodule in this scan exhibits a non-solid appearance with ground-glass opacity",
        2: "The nodule in this scan exhibits a non-solid, mixed appearance",
        3: "The nodule in this scan exhibits a part-solid, mixed appearance",
        4: "The nodule in this scan exhibits a solid, mixed appearance",
        5: "The nodule in this scan exhibits a solid appearance"
    }
    questions = {
        1: "Does the nodule in this scan have a non-solid appearance with ground-glass opacity?",
        2: "Does the nodule in this scan have a non-solid, mixed appearance?",
        3: "Does the nodule in this scan have a part-solid, mixed appearance?",
        4: "Does the nodule in this scan have a solid, mixed appearance?",
        5: "Does the nodule in this scan have a solid appearance?"
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "What radiographic solidity does the nodule in this scan show?",
            "answer": answers[majority_rating],
            "content_type": "texture",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": questions[random_element],
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "texture",
            "answer_type": "CLOSED",
        }


def malignancy_question(ratings: List[int], yes_no: bool):
    category_dict = {
        1: "Highly Unlikely",
        2: "Moderately Unlikely",
        3: "Indeterminate",
        4: "Moderately Suspicious",
        5: "Highly Suspicious"
    }
    majority_rating = get_majority_rating(ratings)
    if majority_rating is None:
        return None

    if not yes_no:
        return {
            "question": "How likely is the nodule in this scan malignant?",
            "answer": f"{category_dict[majority_rating]}",
            "content_type": "malignancy",
            "answer_type": "OPEN",
        }
    else:
        category_ints = list(category_dict.keys())
        probabilities = [0.5 if c == majority_rating else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the nodule in this image {category_dict[random_element].lower()} to be malignant?",
            "answer": "Yes" if random_element == majority_rating else "No",
            "content_type": "malignancy",
            "answer_type": "CLOSED",
        }

def generate_lidc_data(lidc_path):
    metadata = pd.read_csv(lidc_path / "metadata.csv")
    annotation_data = pd.read_csv(lidc_path / "annotation_data.csv")
    print(annotation_data.head())
    features_to_analyze = ["subtlety", "internal Structure", "calcification", "sphericity", "margin", "lobulation",
                           "spiculation", "texture", "malignancy"]
    for column in annotation_data[features_to_analyze]:
        annotation_data[column] = annotation_data[column].apply(lambda x: ast.literal_eval(x))
        # annotation_data[column] = annotation_data[column].apply(
        #     lambda ratings: None if "None" in str(ratings) else ratings
        # )
    questions = []
    qid = 0
    for _, annotation in annotation_data.iterrows():
        manufacturer_df = metadata.loc[metadata["Series UID"] == annotation["Series Instance UID"]]
        assert len(manufacturer_df) == 1
        base_dict = {
            "dicom_series_uid": annotation["Series Instance UID"],
            "patient_id": annotation["Patient ID"],
            "image_file_name": annotation["Image File Name"],
            "scan_id": annotation["Scan ID"],
            "nodule_index": annotation["Nodule Index"],
            "manufacturer": manufacturer_df.iloc[0]["Manufacturer"]
        }

        questions_scan = []
        yes_no = random.choices([True, False])[0]
        questions_scan.append(subtlety_question(list(annotation["subtlety"]), yes_no=yes_no))
        # remove this part
        # yes_no = random.choices([True, False])[0]
        # questions_scan.append(internal_structure_question(list(annotation["internal Structure"]), yes_no=yes_no))
        yes_no = random.choices([True, False])[0]
        questions_scan.append(calcification_question(list(annotation["calcification"]), yes_no=yes_no))
        # We do not use sphericity since we only have 2D images
        yes_no = random.choices([True, False])[0]
        questions_scan.append(margin_question(list(annotation["margin"]), yes_no=yes_no))
        yes_no = random.choices([True, False])[0]
        questions_scan.append(lobulation_question(list(annotation["lobulation"]), yes_no=yes_no))
        yes_no = random.choices([True, False])[0]
        questions_scan.append(spiculation_question(list(annotation["spiculation"]), yes_no=yes_no))
        yes_no = random.choices([True, False])[0]
        questions_scan.append(texture_question(list(annotation["texture"]), yes_no=yes_no))
        yes_no = random.choices([True, False])[0]
        questions_scan.append(malignancy_question(list(annotation["malignancy"]), yes_no=yes_no))

        for question in questions_scan:
            if question is not None:
                print({**base_dict, **question, "qid": qid})
                questions.append({**base_dict, **question, "qid": qid})
                qid += 1

    all_questions = pd.DataFrame(questions)
    all_questions.to_json(lidc_path / "lidc_questions.json", orient="records", lines=False, indent=4)


if __name__ == "__main__":
    cli_args = main_cli()
    if cli_args.path is None:
        path = os.getenv("DATASET_ROOT_DIR")
        assert path is not None
        path = Path(path) / "LIDC"
    else:
        path = Path(cli_args.path)
    random.seed(cli_args.seed)
    print(f"Random seed: {cli_args.seed}")
    generate_lidc_data(lidc_path=path)
