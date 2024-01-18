from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from med_vlm_robustness.dataset import SlakeDataset


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


def get_datamodule(name:str, batch_size:int):
    dataroot = Path("/nvme/VLMRobustness")
    dataset = "Slake"
    data_dir = dataroot / dataset
    identifier = name.split("_")[1:]
    split = identifier[0]
    mode = identifier[1]
    split_category = identifier[2].replace("-", "_")
    split_value = identifier[3].capitalize() if split_category == "content_type" else identifier[3]

    df = pd.read_json(data_dir / f"{mode}.json")
    df = df.loc[df['q_lang'] == "en"]

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
            df_train = df_train.loc[df_train[split_category] == split_value]
            df_val = pd.read_json(data_dir / "validate.json")
            df_val = df_val.loc[df_val[split_category] == split_value]
            df = pd.concat([df_test, df_train, df_val])
    # TODO: pass batch size as argument
    return SlakeDatamodule(data_dir=data_dir, batch_size=batch_size, df=df)
