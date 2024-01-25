import json
import os.path
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import SlakeDataset


class SlakeDatamodule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, df, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = df

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SlakeDataset(self.data_dir, self.df)
        self.val_dataset = SlakeDataset(self.data_dir, self.df)
        self.test_dataset = SlakeDataset(self.data_dir, self.df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def get_slake_df(data_dir, mode, split, split_category=None, split_value=None):
    split_value = split_value.capitalize() if split_category == "content_type" else split_value

    df = pd.read_json(data_dir / f"{mode}.json")
    df = df.loc[df['q_lang'] == "en"]

    if split == "all":
        return df

    if mode == "train" or mode == "validate" or (mode == "test" and split == "iid"):
        df = df.loc[df[split_category] != split_value]
    elif mode == "test" and split == "ood":
        df_test = df.loc[df[split_category] == split_value]
        # If the split value changes within one patient, we only filter the test set,
        # since otherwise we might have the same patient / image within training and test set
        if split_category in ["answer_type", "content_type"]:
            df = df_test
        # If the split values stays constant within one patient, we can also take the training and validation set
        # into the ood test set since this does not imply having the same patient in training / test set
        # TODO: should we really do it like this?
        else:
            df_train = pd.read_json(data_dir / "train.json")
            df_train = df_train.loc[df_train['q_lang'] == "en"]
            df_train = df_train.loc[df_train[split_category] == split_value]
            df_val = pd.read_json(data_dir / "validate.json")
            df_val = df_val.loc[df_val['q_lang'] == "en"]
            df_val = df_val.loc[df_val[split_category] == split_value]
            df = pd.concat([df_test, df_train, df_val])
    return df


def get_datamodule(data_dir:Path, name:str, batch_size:int):
    json_file = get_json_filename(data_dir, name)
    df = pd.read_json(json_file)
    dataset = name.split("_")[0]
    if dataset == "slake":
        return SlakeDatamodule(data_dir=data_dir, batch_size=batch_size, df=df)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


def convert_raw_to_final(df, save_path):
    final_data = []

    # Process each entry as a separate conversation
    for _, row in df.iterrows():
        qid = row["qid"]
        new_entry = {
            "id": str(qid),
            "image": f"imgs/" + row["img_name"],
            "conversations": [
                {
                    "from": "human",
                    "value": row["question"]
                },
                {
                    "from": "gpt",
                    "value": row["answer"]
                }
            ],
            "img_id": row["img_id"],
            "language": row["q_lang"],
            "location": row["location"],
            "modality": row["modality"],
            "answer_type": row["answer_type"],
            "base_type": row["base_type"],
            "content_type": row["content_type"],
        }
        final_data.append(new_entry)

    with open(str(save_path), 'w') as output_file:
        json.dump(final_data, output_file, indent=4)


def get_json_filename(data_dir:Path, name:str):
    identifier = name.split("_")
    dataset = identifier[0]
    if os.path.isfile(data_dir / "split_files" / f"{name}.json"):
        return data_dir / "split_files" / f"{name}.json"
    else:
        mode = identifier[1]
        split = identifier[2]
        if split != "all":
            split_category = identifier[3].replace("-", "_")
            split_value = identifier[4]
        else:
            split_category = None
            split_value = None
        if dataset == "slake":
            df = get_slake_df(data_dir=data_dir, mode=mode, split=split, split_category=split_category,
                              split_value=split_value)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        convert_raw_to_final(df, data_dir / "split_files" / f"{name}.json")
        return data_dir / "split_files" / f"{name}.json"
