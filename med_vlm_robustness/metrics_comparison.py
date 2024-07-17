import json
import os
from itertools import product

import numpy as np
import pandas as pd

from med_vlm_robustness.utils import get_config


def compare_metrics(cfg):
    base_dir = cfg.base_dir
    dataset = cfg.dataset
    model_type = cfg.model_type
    train_split = cfg.train_split
    test_split = cfg.test_split
    mod = cfg.mod
    hyperparams_compare = cfg.hyperparams_compare
    hyperparams_summarize = cfg.hyperparams_summarize

    result_dict = {}
    for params in product(*hyperparams_compare.values()):
        hyperparams = dict(zip(hyperparams_compare.keys(), params))
        hyperparams_compare_str = "_".join([f"{key}{value}" for key, value in hyperparams.items()])
        if hyperparams_compare_str not in result_dict:
            result_dict[hyperparams_compare_str] = {
                "accuracy": [],
                "mistral": [],
            }
        for summarize in product(*hyperparams_summarize.values()):
            hyperparams = dict(zip(hyperparams_compare.keys(), params))
            hyperparams.update(dict(zip(hyperparams_summarize.keys(), summarize)))
            hparams_model_name = "_".join([f"{key}{value}" for key, value in hyperparams.items()])

            results_dir = f"{base_dir}/{dataset}/{model_type}/llava-{dataset}_train_{train_split}-finetune_{model_type}_{hparams_model_name}/eval/{dataset}_{mod}_{test_split}"
            closed_ended_metrics_file = f"{results_dir}/mistral_metrics_closed.json"
            if not os.path.isfile(closed_ended_metrics_file):
                continue
            with open(closed_ended_metrics_file, "r") as f:
                closed_ended_metrics = json.load(f)
            closed_ended_score = closed_ended_metrics[0]["avg_mistral_score"]
            mistral_metrics = f"{results_dir}/mistral_metrics.json"
            with open(mistral_metrics, "r") as f:
                mistral_metrics = json.load(f)
            mistral_score = mistral_metrics[0]["avg_mistral_score"]

            result_dict[hyperparams_compare_str]["accuracy"].append(closed_ended_score)
            result_dict[hyperparams_compare_str]["mistral"].append(mistral_score)

    print(result_dict)
    for key, value in result_dict.items():
        if len(value["accuracy"]) < 3:
            if len(value["accuracy"]) > 0:
                print(f"WARNING: less than 3 (but more than 0) values for: {key}, skipping")
            result_dict[key]["accuracy"] = "NaN"
            result_dict[key]["mistral"] = "NaN"
            continue
        result_dict[key]["accuracy"] = f"{round(np.mean(np.array(value['accuracy'])), 2)} +/- {round(np.std(np.array(value['accuracy']), ddof=1), 2)}"
        result_dict[key]["mistral"] = f"{round(np.mean(np.array(value['mistral'])), 2)} +/- {round(np.std(np.array(value['mistral']), ddof=1), 2)}"
    df = pd.DataFrame(result_dict).transpose()
    results_file = f"{base_dir}/{dataset}/{model_type}/{cfg.output_file_name}"
    df.to_csv(results_file)


if __name__=="__main__":
    config = get_config()
    compare_metrics(cfg=config)
