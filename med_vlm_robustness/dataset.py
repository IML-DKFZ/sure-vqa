from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import random


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
            if answer in ["Yes", "No", "yes", "no"]:
                question += " Please choose from the following two options: [yes, no]."
            # TODO: with llm eval we should not include this, right?
            # TODO : code below should be used to make sure that categorical questions are not counted as CLOSED
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

class SlakeCorruptionDataset(Dataset):
    def __init__(self, dataset_path: Path, json: pd.DataFrame, corruption_probabilities: dict = {'blur':0,'brightness': 0,'noise': 0}, corruption_strength: dict = {'blur':'medium','brightness': 'medium','noise': 'medium'}):
        self.dataset_path = dataset_path
        self.json = json.reset_index(drop=True)
        self.ids = self.json["id"].to_list()
        self.corruption_probabilities = corruption_probabilities
        self.corruption_strength = corruption_strength
        self.corruption_strength_blur = {
            'low' : 5,
            'medium' : 7,
            'high' : 11,
        }
        self.corruption_strength_brightness = {
            'low' : [1.1, 2],
            'medium' : [2.5, 4],
            'high' : [4.5, 6],
        }
        self.corruption_strength_noise = {
            'low' : [0, 0.06],
            'medium' : [0.09, 0.15],
            'high' : [0.18, 0.25],
        }

        self.brightness_factor = None
        self.mean = None
        self.kernel_size = None

    def __len__(self):
        return len(self.ids)

    def apply_corruptions(self, image):
        """Applies multiple corruptions based on the `corruption_probabilities`."""

        self.brightness_factor = None
        self.mean = None
        self.kernel_size = None

        if random.random() < self.corruption_probabilities['blur']:
            self.kernel_size = self.corruption_strength_blur[self.corruption_strength['blur']]
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

        if random.random() < self.corruption_probabilities['brightness']:
            brightness_min = self.corruption_strength_brightness[self.corruption_strength['brightness']][0]
            brightness_max = self.corruption_strength_brightness[self.corruption_strength['brightness']][1]
            self.brightness_factor = random.uniform(brightness_min, brightness_max)  # Always increase brightness but with different strength
            image = cv2.convertScaleAbs(image, alpha=self.brightness_factor, beta=0)

        if random.random() < self.corruption_probabilities['noise']:
            mean_min = self.corruption_strength_noise[self.corruption_strength['noise']][0]
            mean_max = self.corruption_strength_noise[self.corruption_strength['noise']][1]
            self.mean = random.uniform(mean_min, mean_max)
            var = 0.01
            sigma = var ** 0.5
            # Gaussian noise in grayscale
            gaussian = np.random.normal(self.mean, sigma, (image.shape[0], image.shape[1],1))
            image = np.clip(image + gaussian * 255, 0, 255).astype(np.uint8)
        
        if self.brightness_factor == self.mean == self.kernel_size == None:
            random_selection = random.choice(['blur', 'brightness', 'noise'])

            if random_selection == 'blur':
                self.kernel_size = self.corruption_strength_blur[self.corruption_strength['blur']]
                image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

            elif random_selection == 'brightness':
                brightness_min = self.corruption_strength_brightness[self.corruption_strength['brightness']][0]
                brightness_max = self.corruption_strength_brightness[self.corruption_strength['brightness']][1]
                self.brightness_factor = random.uniform(brightness_min, brightness_max)  # Always increase brightness but with different strength
                image = cv2.convertScaleAbs(image, alpha=self.brightness_factor, beta=0)

            elif random_selection == 'noise':
                mean_min = self.corruption_strength_noise[self.corruption_strength['noise']][0]
                mean_max = self.corruption_strength_noise[self.corruption_strength['noise']][1]
                self.mean = random.uniform(mean_min, mean_max)
                var = 0.01
                sigma = var ** 0.5
                # Gaussian noise in grayscale
                gaussian = np.random.normal(self.mean, sigma, (image.shape[0], image.shape[1],1))
                image = np.clip(image + gaussian * 255, 0, 255).astype(np.uint8)

        return image

    def __getitem__(self, index):
        row = self.json.iloc[index]
        image = cv2.imread(str(self.dataset_path / row.image))  # Read image as grayscale

        # Apply the selected corruptions
        image_corrupt = self.apply_corruptions(image)
        
        # Extract question and answer
        question = [i["value"] for i in row.conversations if i["from"] == "human"][0]
        answer = [i["value"] for i in row.conversations if i["from"] == "gpt"][0]
        answer_type = row.answer_type
        
        # Handle closed questions
        if answer_type == "CLOSED":
            if answer in ["Yes", "No", "yes", "no"]:
                question += " Please choose from the following two options: [yes, no]."

        batch = {
            "image_original": image,
            "image": image_corrupt,
            "question": question,
            "gt": answer,
            "qid": row.id,
            "answer_type": answer_type,
            "img_name": row.image,
            # "brightness_factor": self.brightness_factor if self.brightness_factor != None else 0,
            # 'mean': self.mean if self.mean != None else 0,
            # 'kernel_size': self.kernel_size if self.kernel_size != None else 0,
        }
        return batch