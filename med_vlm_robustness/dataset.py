from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset


class SlakeDataset(Dataset):
    def __init__(self, dataset_path: Path, json: pd.DataFrame):
        self.dataset_path = dataset_path
        self.json = json.reset_index(drop=True)
        self.ids = self.json["id"].to_list()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row = self.json.iloc[index]
        image = cv2.imread(str(self.dataset_path / row.image))
        # We assume that there is always only one q/a turn in the conversation
        question = [i["value"] for i in row.conversations if i["from"] == "human"][0]
        answer = [i["value"] for i in row.conversations if i["from"] == "gpt"][0]
        answer_type = row.answer_type
        if answer_type == "CLOSED":
            if answer in ["Yes", "No"]:
                question += " Please choose from the following two options: [yes, no]."
            # TODO: with llm eval we should not include this, right?
            # else:
            #     answer_type = "OPEN"
        batch = {
            "image": image,
            "question": question,
            "gt": answer,
            "qid": row.id,
            "answer_type": answer_type,
            "img_name": row.image,
        }
        return batch