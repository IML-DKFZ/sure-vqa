import argparse
import json
import collections
import random
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from eval.eval_metrics import calculate_exactmatch, calculate_f1score, bleu, \
    calculate_appearance_with_normalization
from eval.glossary import *

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
        if gt in pred:
            yes_no_acc = 1
        else:
            yes_no_acc = 0
        # TODO: discuss if this males sense
        # if 'yes' in pred or 'no' in pred:
        #     if gt in pred:
        #         yes_no_acc = 1
        #     else:
        #         yes_no_acc = 0
        # else:
        #     # yes_no_acc = 0
        return {
            "yes/no accuracy": yes_no_acc
        }

    else:
        exact_score = calculate_exactmatch(pred, gt)
        f1_score, precision, recall = calculate_f1score(pred, gt)
        # b_score = sentence_bleu(references=[str(gt).lower().split()],
        #                         hypothesis=str(pred).lower().split())
        # b_score_1 = sentence_bleu(references=[str(gt).lower().split()],
        #                           hypothesis=str(pred).lower().split(), weights=(1, 0, 0, 0))
        # b_score_2 = sentence_bleu(references=[str(gt).lower().split()],
        #                           hypothesis=str(pred).lower().split(), weights=(0, 1, 0, 0))
        # b_score_3 = sentence_bleu(references=[str(gt).lower().split()],
        #                           hypothesis=str(pred).lower().split(), weights=(0, 0, 1, 0))
        # TODO: isnt this the right calculation
        b_score = sentence_bleu(references=[str(gt).lower().split()],
                                hypothesis=str(pred).lower().split(), weights=[1])
        b_score_1 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=[1])
        b_score_2 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=(1/2, 1/2))
        b_score_3 = sentence_bleu(references=[str(gt).lower().split()],
                                    hypothesis=str(pred).lower().split(), weights=(1/3, 1/3, 1/3))
        # Bleu from Llava-Med paper
        # llava_b_score = bleu(pred.lower(), gt.lower(), n=1, weights=[1])
        # llava_b_score_2 = bleu(pred.lower(), gt.lower(), n=2, weights=[1/2,1/2])
        # llava_b_score_3 = bleu(pred.lower(), gt.lower(), n=3, weights=[1/3,1/3,1/3])
        return {
            'exact match score': exact_score,
            'f1 score': f1_score,
            'precision': precision,
            'recall': recall,
            'bleu_score': b_score,
            'bleu_score_1': b_score_1,
            'bleu_score_2': b_score_2,
            'bleu_score_3': b_score_3,
            # "llava_bleu_score": float(llava_b_score),
            # "llava_bleu_score_2": float(llava_b_score_2),
            # "llava_bleu_score_3": float(llava_b_score_3),
        }

def main(cfg):
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
    # sum_llava_bleu=0
    # sum_llava_bleu_2=0
    # sum_llava_bleu_3=0

    pred_df = pd.read_json(cfg.model_output_file)
    results = []

    #iterate dataframe
    for _, row in pred_df.iterrows():
        pred = row['pred']
        gt = row['gt']
        pred = row['pred'] #  TODO: why is this twice???
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
            # sum_llava_bleu += metrics_dict["llava_bleu_score"]
            # sum_llava_bleu_2 += metrics_dict["llava_bleu_score_2"]
            # sum_llava_bleu_3 += metrics_dict["llava_bleu_score_3"]


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
        # "avg_llava_bleu_score":   sum_llava_bleu / max(num_open_qs, 1),
        # "avg_llava_bleu_score_2": sum_llava_bleu_2/ max(num_open_qs, 1),
        # "avg_llava_bleu_score_3": sum_llava_bleu_3/ max(num_open_qs, 1),
        }
    
    if not Path(cfg.metrics_file).parent.is_dir():
        os.makedirs(Path(cfg.metrics_file).parent)
    with open(cfg.metrics_file, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

    if not Path(cfg.averaged_metrics_file).parent.is_dir():
        os.makedirs(Path(cfg.averaged_metrics_file).parent)
    with open(cfg.averaged_metrics_file, 'w') as f:
        json.dump(average_scores, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    config = get_config()
    main(config)