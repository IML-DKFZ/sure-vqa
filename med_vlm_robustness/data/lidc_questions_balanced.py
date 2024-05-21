import ast
import os
from abc import ABC, abstractmethod
from pathlib import Path
import random

import pandas as pd
import sklearn

from med_vlm_robustness.data.lidc_questions import main_cli
from med_vlm_robustness.data.lidc_questions import get_majority_rating
from med_vlm_robustness.utils import set_seed


class BaseFeature(ABC):
    def __init__(self, feature_string, category_dict):
        self.feature_string = feature_string
        self.category_dict = category_dict

    def set_limit(self, limit):
        self.limit = limit

    @abstractmethod
    def get_open_question(self, category_int):
        pass

    @abstractmethod
    def get_closed_question(self, category_int):
        pass


class Subtlety(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            1: "Extremely Subtle",
            2: "Moderately Subtle",
            3: "Fairly Subtle",
            4: "Moderately Obvious",
            5: "Obvious"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "How easy is the nodule in the scan to be detected?",
            "answer": f"The nodule in the scan is {self.category_dict[category_int].lower()} to detect",
            "content_type": "subtlety",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        category_ints = list(self.category_dict.keys())
        probabilities = [0.5 if c == category_int else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the nodule in this scan {self.category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == category_int else "No",
            "content_type": "subtlety",
            "answer_type": "CLOSED",
        }


class Calcification(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            # 1 and 2 are not present in the data
            # 1: "Popcorn",
            # 2: "Laminated",
            3: "Solid",
            # 4 and 5 are only very few cases
            # 4: "Non-central",
            # 5: "Central",
            6: "Absent"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "Which type of calcification is present in the nodule?",
            "answer": f"The nodule shows a {self.category_dict[category_int].lower()} calcification pattern."
                        if category_int != 6 else "There is no calcification present in the nodule.",
            "content_type": "calcification",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        return {
            "question": f"Does the nodule in this scan show calcification?",
            "answer": "Yes" if category_int != 6 else "No",
            "content_type": "calcification",
            "answer_type": "CLOSED",
        }


class Margin(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            1: "Poorly Defined",
            2: "Near Poorly Defined",
            # Changed from original category "Medium Margin"
            3: "Medium Defined",
            4: "Near Sharp",
            5: "Sharp"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "How well-defined is the margin of the nodule shown in this scan?",
            "answer": f"The nodule in this image has a {self.category_dict[category_int].lower()} margin",
            "content_type": "margin",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        category_ints = list(self.category_dict.keys())
        probabilities = [0.5 if c == category_int else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the margin of the nodule in this scan {self.category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == category_int else "No",
            "content_type": "margin",
            "answer_type": "CLOSED",
        }


class Lobulation(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            1: "No Lobulation",
            2: "Nearly No Lobulation",
            3: "Medium Lobulation",
            4: "Near Marked Lobulation",
            5: "Marked Lobulation"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "What degree of lobulation does the nodule in this scan show?",
            "answer": f"The nodule in this scan shows {self.category_dict[category_int].lower()}",
            "content_type": "lobulation",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        category_ints = list(self.category_dict.keys())
        probabilities = [0.5 if c == category_int else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Does the nodule in this scan show {self.category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == category_int else "No",
            "content_type": "lobulation",
            "answer_type": "CLOSED",
        }


class Spiculation(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            1: "No Spiculation",
            2: "Nearly No Spiculation",
            3: "Medium Spiculation",
            4: "Near Marked Spiculation",
            5: "Marked Spiculation"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "What extent of spiculation does the nodule in this scan show?",
            "answer": f"The nodule in this scan shows {self.category_dict[category_int].lower()}",
            "content_type": "spiculation",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        category_ints = list(self.category_dict.keys())
        probabilities = [0.5 if c == category_int else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Does the nodule in this scan show {self.category_dict[random_element].lower()}?",
            "answer": "Yes" if random_element == category_int else "No",
            "content_type": "spiculation",
            "answer_type": "CLOSED",
        }


class Texture(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            1: "Non-Solid/GGO",
            # 2 are only very few cases
            # 2: "Non-Solid/Mixed",
            3: "Part Solid/Mixed",
            4: "Solid/Mixed",
            5: "Solid"
        }
        self.answers = {
            1: "The nodule in this scan exhibits a non-solid appearance with ground-glass opacity",
            # 2 are only very few cases
            # 2: "The nodule in this scan exhibits a non-solid, mixed appearance",
            3: "The nodule in this scan exhibits a part-solid, mixed appearance",
            4: "The nodule in this scan exhibits a solid, mixed appearance",
            5: "The nodule in this scan exhibits a solid appearance"
        }
        self.questions = {
            1: "Does the nodule in this scan have a non-solid appearance with ground-glass opacity?",
            # 2 are only very few cases
            # 2: "Does the nodule in this scan have a non-solid, mixed appearance?",
            3: "Does the nodule in this scan have a part-solid, mixed appearance?",
            4: "Does the nodule in this scan have a solid, mixed appearance?",
            5: "Does the nodule in this scan have a solid appearance?"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "What radiographic solidity does the nodule in this scan show?",
            "answer": self.answers[category_int],
            "content_type": "texture",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        category_ints = list(self.category_dict.keys())
        probabilities = [0.5 if c == category_int else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": self.questions[random_element],
            "answer": "Yes" if random_element == category_int else "No",
            "content_type": "texture",
            "answer_type": "CLOSED",
        }


class Malignancy(BaseFeature):
    def __init__(self, feature_string):
        category_dict = {
            1: "Highly Unlikely",
            2: "Moderately Unlikely",
            3: "Indeterminate",
            4: "Moderately Suspicious",
            5: "Highly Suspicious"
        }
        super().__init__(feature_string=feature_string, category_dict=category_dict)

    def get_open_question(self, category_int):
        return {
            "question": "How likely is the nodule in this scan malignant?",
            "answer": f"{self.category_dict[category_int]}",
            "content_type": "malignancy",
            "answer_type": "OPEN",
        }

    def get_closed_question(self, category_int):
        category_ints = list(self.category_dict.keys())
        probabilities = [0.5 if c == category_int else 0.5 / (len(category_ints) - 1) for c in category_ints]
        random_element = random.choices(category_ints, weights=probabilities)[0]
        return {
            "question": f"Is the nodule in this image {self.category_dict[random_element].lower()} to be malignant?",
            "answer": "Yes" if random_element == category_int else "No",
            "content_type": "malignancy",
            "answer_type": "CLOSED",
        }


def get_base_dict(annotation, metadata):
    manufacturer_df = metadata.loc[metadata["Series UID"] == annotation["Series Instance UID"]]
    assert len(manufacturer_df) == 1
    return {
        "dicom_series_uid": annotation["Series Instance UID"],
        "patient_id": annotation["Patient ID"],
        "image_file_name": annotation["Image File Name"],
        "scan_id": annotation["Scan ID"],
        "nodule_index": annotation["Nodule Index"],
        "manufacturer": manufacturer_df.iloc[0]["Manufacturer"]
    }


def get_features_majority_dict(annotation):
    features_to_analyze = ["subtlety", "calcification", "margin", "lobulation",
                           "spiculation", "texture", "malignancy"]
    return {f"{feature}_majority": annotation[f"{feature}_majority"] for feature in features_to_analyze}


def get_category_questions(feature: BaseFeature, annotation_data_filtered, metadata):
    annotation_data_filtered = sklearn.utils.shuffle(annotation_data_filtered)
    limit = feature.limit
    if len(annotation_data_filtered) >= limit:
        if len(annotation_data_filtered) > limit:
            annotation_data_filtered = annotation_data_filtered[:limit]
            over_undersampled = "undersampled"
        else:
            over_undersampled = "neither over nor undersampled"
        random_question_type = [True] * len(annotation_data_filtered)
    elif len(annotation_data_filtered) <= limit // 2:
        random_question_type = [False] * len(annotation_data_filtered)
        over_undersampled = "oversampled (doubled)"
    else:
        random_question_type = [False] * (limit - len(annotation_data_filtered)) + [True] * len(annotation_data_filtered)
        over_undersampled = "oversampled"
    questions = []
    for index, annotation in annotation_data_filtered.reset_index().iterrows():
        if random_question_type[index]:
            question_type = random.choice(["open", "closed"])
        else:
            question_type = "both"
        base_dict = get_base_dict(annotation, metadata)
        majority_rating = annotation[f"{feature.feature_string}_majority"]
        if question_type == "open":
            questions.append({**base_dict, **feature.get_open_question(majority_rating),
                              **get_features_majority_dict(annotation)})
        elif question_type == "closed":
            questions.append({**base_dict, **feature.get_closed_question(majority_rating),
                              **get_features_majority_dict(annotation)})
        elif question_type == "both":
            questions.extend([{**base_dict, **feature.get_open_question(majority_rating),
                               **get_features_majority_dict(annotation)},
                              {**base_dict, **feature.get_closed_question(majority_rating),
                               **get_features_majority_dict(annotation)}])
        else:
            raise ValueError(f"Invalid question type: {question_type}")
    return questions, over_undersampled


def get_feature_questions(annotation_data, metadata, feature_string, limit_class: int, double_limit_class: bool = True):
    feature = feature_dict[feature_string](feature_string)
    limit = len(annotation_data.loc[annotation_data[f"{feature.feature_string}_majority"] == limit_class])
    if double_limit_class:
        limit *= 2
    feature.set_limit(limit)
    print("===============================")
    print(f"{feature_string.upper()}, Limit of questions per category: {limit} (based on class {limit_class}, "
          f"{'class doubled' if double_limit_class else 'class not doubled'})")
    feature_questions = []
    for i in feature.category_dict.keys():
        annotations = annotation_data.loc[annotation_data[f"{feature.feature_string}_majority"] == i]
        q, over_undersampled = get_category_questions(feature, annotations, metadata)
        print(f"{feature.feature_string} class {i}: {len(q)} ({over_undersampled})")
        feature_questions.extend(q)
    print(f"Overall questions for {feature_string}: {len(feature_questions)}")
    return feature_questions


def generate_lidc_data(lidc_path):
    metadata = pd.read_csv(lidc_path / "metadata.csv")
    annotation_data = pd.read_csv(lidc_path / "annotation_data.csv")
    # We do not use internal structure since it is too unbalanced
    # We do not use sphericity since we only have 2D images
    features_to_analyze = ["subtlety", "calcification", "margin", "lobulation",
                           "spiculation", "texture", "malignancy"]
    for column in annotation_data[features_to_analyze]:
        annotation_data[column] = annotation_data[column].apply(lambda x: ast.literal_eval(x))
    for feature in features_to_analyze:
        annotation_data[f"{feature}_majority"] = annotation_data[feature].apply(lambda x: get_majority_rating(list(x)))
    all_questions = []
    for feature in features_to_analyze:
        all_questions.extend(get_feature_questions(annotation_data, metadata, feature_string=feature,
                              limit_class=feature_limit_class_dict[feature][0],
                              double_limit_class=feature_limit_class_dict[feature][1]))
    all_questions = pd.DataFrame(all_questions)
    all_questions.index.name = "qid"
    all_questions.reset_index(inplace=True)
    print("===============================")
    print("===============================")
    print(f"Number of all questions: {len(all_questions)}")
    all_questions.to_json(lidc_path / "lidc_questions.json", orient="records", lines=False, indent=4)


feature_dict = {
    "subtlety": Subtlety,
    "calcification": Calcification,
    "margin": Margin,
    "lobulation": Lobulation,
    "spiculation": Spiculation,
    "texture": Texture,
    "malignancy": Malignancy
}

feature_limit_class_dict = {
    "subtlety": (2, True),
    "calcification": (3, True),
    "margin": (2, True),
    "lobulation": (4, True),
    "spiculation": (3, True),
    "texture": (1, False),
    "malignancy": (1, False)
}

if __name__ == "__main__":
    cli_args = main_cli()
    if cli_args.path is None:
        path = os.getenv("DATASET_ROOT_DIR")
        assert path is not None
        path = Path(path) / "LIDC"
    else:
        path = Path(cli_args.path)
    random.seed(cli_args.seed)
    set_seed(cli_args.seed)

    print(f"Random seed: {cli_args.seed}")
    generate_lidc_data(lidc_path=path)