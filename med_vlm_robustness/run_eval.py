import argparse
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from eval.eval_metrics import calculate_exactmatch, calculate_f1score, get_accuracy, get_open_ended_metrics
from eval.glossary import *
from mistral_eval import mistal_eval, average_mistral_metrics

from pathlib import Path
from utils import get_config
import os
import warnings

warnings.simplefilter('ignore')


def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--pred', type=str, help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args


def evaluate(gt, pred, answer_type):
    gt = gt.lower()
    pred = pred.lower()

    gt = normalize_word(gt)
    pred = normalize_word(pred)

    if answer_type == "CLOSED":
        # for close-ended question (Yes/No)
        if (gt in pred) or (pred in gt) and len(pred) != 0:
            yes_no_acc = 1
        else:
            yes_no_acc = 0
        return {
            "yes/no accuracy": yes_no_acc
        }

    else:
        exact_score = calculate_exactmatch(pred, gt)
        f1_score, precision, recall = calculate_f1score(pred, gt)
        b_score = sentence_bleu(references=[str(gt).lower().split()],
                                hypothesis=str(pred).lower().split(), weights=[1])
        b_score_1 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=[1])
        b_score_2 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=(1/2, 1/2))
        b_score_3 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=(1/3, 1/3, 1/3))
        return {
            'exact match score': exact_score,
            'f1 score': f1_score,
            'precision': precision,
            'recall': recall,
            'bleu_score': b_score,
            'bleu_score_1': b_score_1,
            'bleu_score_2': b_score_2,
            'bleu_score_3': b_score_3,
        }


def main_old(cfg):
    # set the params to calculate the average
    num_closed_qs=0
    num_open_qs=0
    sum_yes_no_acc=0
    sum_exact_match_score=0
    sum_f1_score=0
    sum_prec=0
    sum_recall=0
    sum_bleu=0
    sum_bleu_1=0
    sum_bleu_2=0
    sum_bleu_3=0

    pred_df = pd.read_json(cfg.model_output_file)
    results = []

    #iterate dataframe
    for _, row in pred_df.iterrows():
        pred = row['pred']
        gt = row['gt']
        answer_type = row['answer_type']
        metrics_dict = evaluate(gt=gt, pred=pred, answer_type=answer_type)

        # TODO: TEST this
        if answer_type == "CLOSED":
            num_closed_qs += 1
            sum_yes_no_acc += metrics_dict["yes/no accuracy"]
        else:
            num_open_qs += 1
            sum_exact_match_score += metrics_dict['exact match score']
            sum_f1_score += metrics_dict['f1 score']
            sum_prec += metrics_dict['precision']
            sum_recall += metrics_dict['recall']
            sum_bleu += metrics_dict['bleu_score']
            sum_bleu_1 += metrics_dict['bleu_score_1']
            sum_bleu_2 += metrics_dict['bleu_score_2']
            sum_bleu_3 += metrics_dict['bleu_score_3']


        results.append({
            "qid": row['qid'],
            "answer_type": answer_type,
            "metrics": metrics_dict,
        })

    average_scores = {
        'avg_yes_no_acc': sum_yes_no_acc / max(num_closed_qs,1), # if num_closed_qs = 0, set it to 1 otherwise division by zero error 
        'avg_exact match score': sum_exact_match_score / max(num_open_qs, 1),
        'avg_f1 score': sum_f1_score / max(num_open_qs, 1),
        'avg_precision': sum_prec / max(num_open_qs, 1),
        'avg_recall': sum_recall / max(num_open_qs, 1),
        'avg_bleu_score': sum_bleu / max(num_open_qs, 1),
        'avg_bleu_score_1': sum_bleu_1 / max(num_open_qs, 1),
        'avg_bleu_score_2': sum_bleu_2 / max(num_open_qs, 1),
        'avg_bleu_score_3':  sum_bleu_3   / max(num_open_qs, 1),
        }
    
    if not Path(cfg.metrics_file).parent.is_dir():
        os.makedirs(Path(cfg.metrics_file).parent)
    with open(cfg.metrics_file, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

    if not Path(cfg.averaged_metrics_file).parent.is_dir():
        os.makedirs(Path(cfg.averaged_metrics_file).parent)
    with open(cfg.averaged_metrics_file, 'w') as f:
        json.dump(average_scores, f, indent=4, sort_keys=True)

def evaluate_open_ended(df):
    # set the params to calculate the average
    num_open_qs=0
    sum_exact_match_score=0
    sum_f1_score=0
    sum_prec=0
    sum_recall=0
    sum_bleu=0
    sum_bleu_1=0
    sum_bleu_2=0
    sum_bleu_3=0

    results = [{
        'avg_exact_match_score': 0,
        'avg_f1_score': 0,
        'avg_precision': 0,
        'avg_recall': 0,
        'avg_bleu_score': 0,
        'avg_bleu_score_1': 0,
        'avg_bleu_score_2': 0,
        'avg_bleu_score_3': 0,
    }]
    for _, row in df.iterrows():
        pred = row['pred'].lower()
        gt = row['gt'].lower()
        answer_type = row['answer_type']
        
        if answer_type == 'OPEN':
            metrics = get_open_ended_metrics(gt, pred)
            num_open_qs += 1
            sum_exact_match_score += metrics['exact_match_score']
            sum_f1_score += metrics['f1_score']
            sum_prec += metrics['precision']
            sum_recall += metrics['recall']
            sum_bleu += metrics['bleu_score']
            sum_bleu_1 += metrics['bleu_score_1']
            sum_bleu_2 += metrics['bleu_score_2']
            sum_bleu_3 += metrics['bleu_score_3']
            row['exact_match_score'] = metrics['exact_match_score']
            row['f1_score'] = metrics['f1_score']
            row['precision'] = metrics['precision']
            row['recall'] = metrics['recall']
            row['bleu_score'] = metrics['bleu_score']
            row['bleu_score_1'] = metrics['bleu_score_1']
            row['bleu_score_2'] = metrics['bleu_score_2']
            row['bleu_score_3'] = metrics['bleu_score_3']
            
        results.append(row.to_dict())
    
    results[0]['avg_exact_match_score'] = sum_exact_match_score / max(num_open_qs, 1)
    results[0]['avg_f1_score'] = sum_f1_score / max(num_open_qs, 1)
    results[0]['avg_precision'] = sum_prec / max(num_open_qs, 1)
    results[0]['avg_recall'] = sum_recall / max(num_open_qs, 1)
    results[0]['avg_bleu_score'] = sum_bleu / max(num_open_qs, 1)
    results[0]['avg_bleu_score_1'] = sum_bleu_1 / max(num_open_qs, 1)
    results[0]['avg_bleu_score_2'] = sum_bleu_2 / max(num_open_qs, 1)
    results[0]['avg_bleu_score_3'] =  sum_bleu_3   / max(num_open_qs, 1)
    
    return results


def get_eval_path(cfg):
    if cfg.split == "all" or cfg.split == "sample":
        split_file_test = f"{cfg.dataset}_{cfg.mod}_{cfg.split}".replace(" ", "")
    else:
        split_category = cfg.data_shift
        split_value = cfg.ood_value
        split_file_test = f"{cfg.dataset}_{cfg.mod}_{cfg.split}_{split_category}_{split_value}".replace(" ", "")
    split_file_train = split_file_test.replace(cfg.mod, 'train').replace('ood', 'iid')
    
    # in case the train split doesnt use sample dataset but inference does
    if 'sample' not in cfg.train_split:
        split_file_train = split_file_train.replace('sample_iid', 'iid')

    if cfg.train_no_image:
        split_file_train = split_file_train + '_no_image'
    if cfg.no_image:
        split_file_test = split_file_test + '_no_image'

    if cfg.corruption:
        split_file_test = split_file_test + '_corruption'
        strength = cfg.corruption_strength['blur']
        split_file_test = split_file_test + '_' + str(strength) 

    # print('CORRUPTION: ', cfg.corruption)
    # print(str(cfg.corruption_strength))

    model_name = f"llava-{split_file_train}-finetune_{cfg.model_type}"
    if "hyperparams_model_name" in cfg and cfg.hyperparams_model_name is not None:
        model_name = f"{model_name}_{cfg.hyperparams_model_name}"
    if cfg.model_type != "pretrained":
        eval_path = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{cfg.dataset}/{cfg.model_type}/{model_name}/eval/{split_file_test}"
    else:
        eval_path = f"{os.getenv('EXPERIMENT_ROOT_DIR')}/{cfg.dataset}/{cfg.model_type}/eval/{split_file_test}"
    return Path(eval_path)


def main(cfg):
    eval_path = get_eval_path(cfg)
    model_output_file = eval_path / "test_results.json"
    pred_df = pd.read_json(model_output_file)
    train_df = pd.read_json(cfg.model_train_file) 
    val_df = pd.read_json(cfg.model_val_file) 
    test_df = pd.read_json(cfg.model_test_file) 

    # combine the dataframes to have one big dataframe containing all the samples with num_categories and list_categories columns
    full_dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)
    pred_df = pred_df[(pred_df['qid'].isin(full_dataset['qid']))] # remove later
                      
    pred_df_closed = pred_df[pred_df['answer_type']=='CLOSED']
    pred_df_open = pred_df[pred_df['answer_type']=='OPEN']

    if "traditional_metrics" in cfg.metric_type:
        open_ended_results = evaluate_open_ended(pred_df_open)
        if cfg.dataset != "MIMIC":
            close_ended_results = get_accuracy(pred_df_closed, full_dataset)
        else:
            if cfg.mod == "train":
                dataset = train_df
            elif cfg.mod == "val":
                dataset = val_df
            else:
                dataset = test_df
            close_ended_results = get_accuracy(pred_df_closed, dataset)

        close_ended_metrics_file = eval_path / "closed_ended_metrics.json"
        if not Path(close_ended_metrics_file).parent.is_dir():
            os.makedirs(Path(close_ended_metrics_file).parent)
        with open(close_ended_metrics_file, 'w') as f:
            json.dump(close_ended_results, f, indent=4, sort_keys=True)

        open_ended_metrics_file = eval_path / "open_ended_metrics.json"
        if not Path(open_ended_metrics_file).parent.is_dir():
            os.makedirs(Path(open_ended_metrics_file).parent)
        with open(open_ended_metrics_file, 'w') as f:
            json.dump(open_ended_results, f, indent=4, sort_keys=True)

    if "mistral" in cfg.metric_type:
        mistral_scores = mistal_eval(model_output_file=model_output_file)
        mistral_metrics_file = eval_path / "mistral_metrics.json"
        if not Path(mistral_metrics_file).parent.is_dir():
            os.makedirs(Path(mistral_metrics_file).parent)
        with open(mistral_metrics_file, 'w') as json_file:
            json.dump(mistral_scores, json_file, indent=4)

    if "mistral_closed" in cfg.metric_type:
        if cfg.dataset != "MIMIC":
            mistral_scores = mistal_eval(model_output_file=model_output_file, closed=True)
        else:
            if cfg.mod == "train":
                dataset = train_df
            elif cfg.mod == "val":
                dataset = val_df
            else:
                dataset = test_df
            mistral_scores = mistal_eval(model_output_file=model_output_file, closed=True, data_categories=dataset)
            mistral_scores_multilabel = mistal_eval(model_output_file=model_output_file, closed=True, multilabel=True,
                                                    data_categories=dataset)
            mistral_scores = [*mistral_scores[1:], *mistral_scores_multilabel[1:]]
            mistral_scores = average_mistral_metrics(mistral_scores, closed=True)
        mistral_metrics_file = eval_path / "mistral_metrics_closed.json"
        if not Path(mistral_metrics_file).parent.is_dir():
            os.makedirs(Path(mistral_metrics_file).parent)
        with open(mistral_metrics_file, 'w') as json_file:
            json.dump(mistral_scores, json_file, indent=4)


if __name__ == '__main__':
    config = get_config()
    main(config)