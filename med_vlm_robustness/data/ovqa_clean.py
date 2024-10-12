import pandas as pd 
import json 
from argparse import ArgumentParser
from pathlib import Path

def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to the OVQA dataset containing train. test and validation files "
             "as described in the README.",
        default=None,
    )

    args = parser.parse_args()
    return args

def get_cleaned_data(df):
    non_yes_no_subset = df[~df['answer'].str.lower().isin(['yes','yes.','no','no.'])] # get the subset of samples having non yes/no questions (because we wanna keep yes no question samples)
    # Keep only the samples having following questions: Is this a CT image or X-Ray image?|Is this an MRI or a CT scan?|Was this image taken with an MRI or CT scanner?
    samples_to_remove = non_yes_no_subset[
        (non_yes_no_subset['answer_type'] == 'CLOSED') & 
        ~(non_yes_no_subset['question'].str.contains(
            "Is this a CT image or X-Ray image?|Is this an MRI or a CT scan?|Was this image taken with an MRI or CT scanner?"
        ))
    ]

    final_df = df[~df['qid'].isin(samples_to_remove['qid'])] # remove the problematic samples

    return final_df

def clean_ovqa(args):

    dataset_path = str(Path(args.path))
    
    train_data = pd.read_json(dataset_path + '/train.json')
    val_data = pd.read_json(dataset_path + '/validate.json')
    test_data = pd.read_json(dataset_path + '/test.json')

    # save df as the old data
    with open(dataset_path + '/train_old.json', 'w') as f:
        json.dump(train_data.to_dict(orient="records"), f, indent=4)
    # save df as the old data
    with open(dataset_path + '/validate_old.json', 'w') as f:
        json.dump(val_data.to_dict(orient="records"), f, indent=4)
    # save df as the old data
    with open(dataset_path + '/test_old.json', 'w') as f:
        json.dump(test_data.to_dict(orient="records"), f, indent=4)

    train_data = get_cleaned_data(train_data)
    val_data = get_cleaned_data(val_data)
    test_data = get_cleaned_data(test_data)

    # save df as the old data
    with open(dataset_path + '/train.json', 'w') as f:
        json.dump(train_data.to_dict(orient="records"), f, indent=4)
    # save df as the old data
    with open(dataset_path + '/validate.json', 'w') as f:
        json.dump(val_data.to_dict(orient="records"), f, indent=4)
    # save df as the old data
    with open(dataset_path + '/test.json', 'w') as f:
        json.dump(test_data.to_dict(orient="records"), f, indent=4)
        

if __name__== "__main__":
    args = main_cli()
    clean_ovqa(args)