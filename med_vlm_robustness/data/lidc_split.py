import os
import random
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to the LIDC dataset. Should contain the lidc_questions.json file. "
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

def create_lidc_split(lidc_path):
    questions = pd.read_json(lidc_path / "lidc_questions.json", dtype=str)
    patients = sorted(list(set(questions["patient_id"])))
    num_train = int(len(patients) * 0.8)
    patients_train = random.sample(patients, num_train)
    patients_test = [p for p in patients if p not in patients_train]

    num_val = int(num_train * 0.25)
    patients_val = random.sample(patients_train, num_val)
    patients_train = [p for p in patients_train if p not in patients_val]

    train_df = questions[questions["patient_id"].isin(patients_train)]
    val_df = questions[questions["patient_id"].isin(patients_val)]
    test_df = questions[questions["patient_id"].isin(patients_test)]

    for patient in train_df["patient_id"]:
        assert patient in patients_train
        assert patient not in patients_val
        assert patient not in patients_test

    for patient in val_df["patient_id"]:
        assert patient in patients_val
        assert patient not in patients_train
        assert patient not in patients_test

    for patient in test_df["patient_id"]:
        assert patient in patients_test
        assert patient not in patients_train
        assert patient not in patients_val

    train_df.to_json(lidc_path / "train.json", orient="records", lines=False, indent=4)
    val_df.to_json(lidc_path / "validate.json", orient="records", lines=False, indent=4)
    test_df.to_json(lidc_path / "test.json", orient="records", lines=False, indent=4)


if __name__=="__main__":
    cli_args = main_cli()
    if cli_args.path is None:
        path = os.getenv("DATASET_ROOT_DIR")
        assert path is not None
        path = Path(path) / "LIDC"
    else:
        path = Path(cli_args.path)
    random.seed(cli_args.seed)
    print(f"Random seed: {cli_args.seed}")
    create_lidc_split(lidc_path=path)
