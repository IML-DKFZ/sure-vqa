import pandas as pd
from evaluate import load
import os
from pathlib import Path
import json
from tqdm import tqdm
from utils import get_config

def get_bert_metrics(cfg):

    bertscore = load("bertscore")
    file = pd.read_json(cfg.model_output_file)

    bert_scores = []
    num_open_qs = 0
    num_yes_no_qs = 0
    yes_no_scores = {
        "sum_bert_precision" : 0,
        "sum_bert_recall" : 0,
        "sum_bert_f1" : 0,
    }
    open_scores = {
        "sum_bert_precision" : 0,
        "sum_bert_recall" : 0,
        "sum_bert_f1" : 0,
    }
    for _, line in tqdm(file.iterrows()):
        qid = line["qid"]
        question = line["question"]
        gt = line["gt"]
        pred = line["pred"]
        answer_type = line["answer_type"]
        pred = line["pred"]
        gt = line["gt"]

        bert_results = bertscore.compute(predictions=[pred], references=[gt], lang="en")

        # create a dict to keep the scores
        output_dict={
            "qid": qid,
            "question": question,
            "gt": gt,
            "pred": pred,
            "answer_type": answer_type,
            "bert_precision" : bert_results["precision"][0],
            "bert_recall" : bert_results["recall"][0],
            "bert_f1" : bert_results["f1"][0]
        }
        bert_scores.append(output_dict)

        if answer_type == "CLOSED":
            num_yes_no_qs += 1
            yes_no_scores["sum_bert_precision"] += bert_results["precision"][0]
            yes_no_scores["sum_bert_recall"] += bert_results["recall"][0]
            yes_no_scores["sum_bert_f1"] += bert_results["f1"][0]
        else:
            num_open_qs += 1
            open_scores["sum_bert_precision"] += bert_results["precision"][0]
            open_scores["sum_bert_recall"] += bert_results["recall"][0]
            open_scores["sum_bert_f1"] += bert_results["f1"][0]
        
    avg_bert_scores = {
        "avg_open_bert_precision": open_scores["sum_bert_precision"] / max(1, num_open_qs),
        "avg_open_bert_recall": open_scores["sum_bert_recall"] / max(1, num_open_qs),
        "avg_open_bert_f1": open_scores["sum_bert_f1"] / max(1, num_open_qs),
        "avg_yes_no_bert_precision": yes_no_scores["sum_bert_precision"] / max(1, num_yes_no_qs),
        "avg_yes_no_bert_recall": yes_no_scores["sum_bert_recall"] / max(1, num_yes_no_qs),
        "avg_yes_no_bert_f1": yes_no_scores["sum_bert_f1"] / max(1, num_yes_no_qs)
    }

    if not Path(cfg.metrics_file).parent.is_dir():
        os.makedirs(Path(cfg.metrics_file).parent)
    with open(cfg.metrics_file, 'w') as f:
        json.dump(bert_scores, f, indent=4, sort_keys=True)
    
    if not Path(cfg.averaged_metrics_file).parent.is_dir():
        os.makedirs(Path(cfg.averaged_metrics_file).parent)
    with open(cfg.averaged_metrics_file, 'w') as f:
        json.dump(avg_bert_scores, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    config = get_config()
    get_bert_metrics(config)