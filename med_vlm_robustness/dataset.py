from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset


class SlakeDataset(Dataset):
    def __init__(self, dataset_path: Path, json: pd.DataFrame):
        self.dataset_path = dataset_path
        self.json = json.reset_index(drop=True)
        self.question = self.json["question"].to_list()
        self.answers = self.json["answer"].to_list()

    def __len__(self):
        return len(self.question)

    def __getitem__(self, index):
        row = self.json.iloc[index]
        image = cv2.imread(str(self.dataset_path / "imgs" / row.img_name))
        question = row.question
        answer = row.answer
        answer_type = row.answer_type
        if answer_type == "CLOSED":
            if answer in ["Yes", "No"]:
                question += " Please choose from the following two options: [yes, no]."
            else:
                answer_type = "OPEN"
        batch = {
            "image": image,
            "question": question,
            "gt": answer,
            "qid": row.qid,
            "answer_type": answer_type,
            "img_name": row.img_name,
        }
        print(row.qid, index)
        return batch