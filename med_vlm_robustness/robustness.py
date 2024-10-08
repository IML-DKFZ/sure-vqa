import json
import os
from itertools import product

import numpy as np
import pandas as pd

from med_vlm_robustness.utils import get_config


def calc_robustness_pretrained(cfg):
    base_dir = cfg.base_dir
    dataset = cfg.dataset
    model_type = cfg.model_type
    mod = cfg.mod

    result_dict_iid = {
        "accuracy": None,
        "mistral": None,
    }
    result_dict_ood = {
        "accuracy": None,
        "mistral": None,
    }
    result_dict_rr = {
        "accuracy": None,
        "mistral": None,
    }

    result_dirs = []
    for test_split in ["iid", "ood"]:
        if not cfg.get("no_image", False):
            result_dirs.append(
                f"{base_dir}/{dataset}/{model_type}/eval/{dataset}_{mod}_{test_split}_{cfg.data_shift}_{cfg.ood_value.replace(' ', '')}")
        else:
            result_dirs.append(
                f"{base_dir}/{dataset}/{model_type}/eval/{dataset}_{mod}_{test_split}_{cfg.data_shift}_{cfg.ood_value.replace(' ', '')}_no_image")
    result_dir_iid = result_dirs[0]
    result_dir_ood = result_dirs[1]
    closed_ended_metrics_file_iid = f"{result_dir_iid}/mistral_metrics_closed.json"
    closed_ended_metrics_file_ood = f"{result_dir_ood}/mistral_metrics_closed.json"
    with open(closed_ended_metrics_file_iid, "r") as f:
        closed_ended_metrics_iid = json.load(f)
    closed_ended_score_iid = closed_ended_metrics_iid[0]["avg_mistral_score"]
    with open(closed_ended_metrics_file_ood, "r") as f:
        closed_ended_metrics_ood = json.load(f)
    closed_ended_score_ood = closed_ended_metrics_ood[0]["avg_mistral_score"]
    mistral_metrics_iid = f"{result_dir_iid}/mistral_metrics.json"
    with open(mistral_metrics_iid, "r") as f:
        mistral_metrics_iid = json.load(f)
    mistral_score_iid = mistral_metrics_iid[0]["avg_mistral_score"]
    mistral_metrics_ood = f"{result_dir_ood}/mistral_metrics.json"
    with open(mistral_metrics_ood, "r") as f:
        mistral_metrics_ood = json.load(f)
    mistral_score_ood = mistral_metrics_ood[0]["avg_mistral_score"]

    rr_closed = 1 - ((closed_ended_score_iid - closed_ended_score_ood) / closed_ended_score_iid)
    rr_mistral = 1 - ((mistral_score_iid - mistral_score_ood) / mistral_score_iid)

    result_dict_iid["accuracy"] = round(closed_ended_score_iid, 2)
    result_dict_iid["mistral"] = round(mistral_score_iid, 2)
    result_dict_ood["accuracy"] = round(closed_ended_score_ood, 2)
    result_dict_ood["mistral"] = round(mistral_score_ood, 2)
    result_dict_rr["accuracy"] = round(rr_closed, 2)
    result_dict_rr["mistral"] = round(rr_mistral, 2)

    print(result_dict_iid)
    print(result_dict_ood)
    print(result_dict_rr)

    result_dict = {
        "iid": result_dict_iid,
        "ood": result_dict_ood,
        "rr": result_dict_rr,
    }
    result_df = pd.DataFrame(result_dict).transpose()
    print(result_df)
    results_file = f"{base_dir}/{dataset}/{model_type}/{cfg.output_file_name}"
    print(results_file)
    result_df.to_csv(results_file)


def calc_robustness(cfg):
    base_dir = cfg.base_dir
    dataset = cfg.dataset
    model_type = cfg.model_type
    train_split = cfg.train_split
    # test_split = cfg.test_split
    mod = cfg.mod
    hyperparams_compare = cfg.hyperparams_compare
    hyperparams_summarize = cfg.hyperparams_summarize

    result_dict_iid = {}
    result_dict_ood = {}
    result_dict_rr = {}
    for params in product(*hyperparams_compare.values()):
        hyperparams = dict(zip(hyperparams_compare.keys(), params))
        hyperparams_compare_str = "_".join([f"{key}{value}" for key, value in hyperparams.items()])
        if hyperparams_compare_str not in result_dict_iid:
            result_dict_iid[hyperparams_compare_str] = {
                "accuracy": [],
                "mistral": [],
            }
        if hyperparams_compare_str not in result_dict_ood:
            result_dict_ood[hyperparams_compare_str] = {
                "accuracy": [],
                "mistral": [],
            }
        if hyperparams_compare_str not in result_dict_rr:
            result_dict_rr[hyperparams_compare_str] = {
                "accuracy": [],
                "mistral": [],
            }
        for summarize in product(*hyperparams_summarize.values()):
            hyperparams = dict(zip(hyperparams_compare.keys(), params))
            hyperparams.update(dict(zip(hyperparams_summarize.keys(), summarize)))
            hparams_model_name = "_".join([f"{key}{value}" for key, value in hyperparams.items()])

            result_dirs = []
            for test_split in ["iid", "ood"]:
                if not cfg.get("no_image", False):
                    result_dirs.append(f"{base_dir}/{dataset}/{model_type}/llava-{dataset}_train_{train_split}_{cfg.data_shift}_{cfg.ood_value.replace(' ', '')}-finetune_{model_type}_{hparams_model_name}/eval/{dataset}_{mod}_{test_split}_{cfg.data_shift}_{cfg.ood_value.replace(' ', '')}")
                else:
                    result_dirs.append(
                        f"{base_dir}/{dataset}/{model_type}/llava-{dataset}_train_{train_split}_{cfg.data_shift}_{cfg.ood_value.replace(' ', '')}_no_image-finetune_{model_type}_{hparams_model_name}/eval/{dataset}_{mod}_{test_split}_{cfg.data_shift}_{cfg.ood_value.replace(' ', '')}_no_image")
            result_dir_iid = result_dirs[0]
            result_dir_ood = result_dirs[1]
            closed_ended_metrics_file_iid = f"{result_dir_iid}/mistral_metrics_closed.json"
            closed_ended_metrics_file_ood = f"{result_dir_ood}/mistral_metrics_closed.json"
            with open(closed_ended_metrics_file_iid, "r") as f:
                closed_ended_metrics_iid = json.load(f)
            closed_ended_score_iid = closed_ended_metrics_iid[0]["avg_mistral_score"]
            with open(closed_ended_metrics_file_ood, "r") as f:
                closed_ended_metrics_ood = json.load(f)
            closed_ended_score_ood = closed_ended_metrics_ood[0]["avg_mistral_score"]
            mistral_metrics_iid = f"{result_dir_iid}/mistral_metrics.json"
            with open(mistral_metrics_iid, "r") as f:
                mistral_metrics_iid = json.load(f)
            mistral_score_iid = mistral_metrics_iid[0]["avg_mistral_score"]
            mistral_metrics_ood = f"{result_dir_ood}/mistral_metrics.json"
            with open(mistral_metrics_ood, "r") as f:
                mistral_metrics_ood = json.load(f)
            mistral_score_ood = mistral_metrics_ood[0]["avg_mistral_score"]

            rr_closed = 1 - ((closed_ended_score_iid - closed_ended_score_ood) / closed_ended_score_iid)
            rr_mistral = 1 - ((mistral_score_iid - mistral_score_ood) / mistral_score_iid)

            result_dict_iid[hyperparams_compare_str]["accuracy"].append(closed_ended_score_iid)
            result_dict_iid[hyperparams_compare_str]["mistral"].append(mistral_score_iid)
            result_dict_ood[hyperparams_compare_str]["accuracy"].append(closed_ended_score_ood)
            result_dict_ood[hyperparams_compare_str]["mistral"].append(mistral_score_ood)
            result_dict_rr[hyperparams_compare_str]["accuracy"].append(rr_closed)
            result_dict_rr[hyperparams_compare_str]["mistral"].append(rr_mistral)

    print(result_dict_iid)
    print(result_dict_ood)
    print(result_dict_rr)
    # result_dict_plot = {key:value.copy() for key, value in result_dict.items() if len(value["accuracy"]) > 0}
    # # result_dict = result_dict_filtered
    for key, value in result_dict_iid.items():
        result_dict_iid[key]["accuracy"] = f"{round(np.mean(np.array(value['accuracy'])), 2)} +/- {round(np.std(np.array(value['accuracy']), ddof=1), 2)}"
        result_dict_iid[key]["mistral"] = f"{round(np.mean(np.array(value['mistral'])), 2)} +/- {round(np.std(np.array(value['mistral']), ddof=1), 2)}"
    for key, value in result_dict_ood.items():
        result_dict_ood[key]["accuracy"] = f"{round(np.mean(np.array(value['accuracy'])), 2)} +/- {round(np.std(np.array(value['accuracy']), ddof=1), 2)}"
        result_dict_ood[key]["mistral"] = f"{round(np.mean(np.array(value['mistral'])), 2)} +/- {round(np.std(np.array(value['mistral']), ddof=1), 2)}"
    for key, value in result_dict_rr.items():
        result_dict_rr[key]["accuracy"] = f"{round(np.mean(np.array(value['accuracy'])), 2)} +/- {round(np.std(np.array(value['accuracy']), ddof=1), 2)}"
        result_dict_rr[key]["mistral"] = f"{round(np.mean(np.array(value['mistral'])), 2)} +/- {round(np.std(np.array(value['mistral']), ddof=1), 2)}"
    result_dict_iid["iid"] = result_dict_iid.pop(list(result_dict_iid.keys())[0])
    result_dict_ood["ood"] = result_dict_ood.pop(list(result_dict_ood.keys())[0])
    result_dict_rr["rr"] = result_dict_rr.pop(list(result_dict_rr.keys())[0])
    df_iid = pd.DataFrame(result_dict_iid).transpose()
    df_ood = pd.DataFrame(result_dict_ood).transpose()
    df_rr = pd.DataFrame(result_dict_rr).transpose()
    result_df = pd.concat([df_iid, df_ood, df_rr], axis=0)
    print(result_df)
    results_file = f"{base_dir}/{dataset}/{model_type}/{cfg.output_file_name}"
    print(results_file)
    result_df.to_csv(results_file)


if __name__=="__main__":
    config = get_config()
    if config.model_type == "pretrained":
        calc_robustness_pretrained(cfg=config)
    else:
        calc_robustness(cfg=config)