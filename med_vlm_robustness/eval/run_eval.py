import argparse
import json
import collections
import random
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics import calculate_exactmatch, calculate_f1score, bleu, \
    calculate_appearance_with_normalization
from glossary import *

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
        if 'yes' in pred or 'no' in pred:
            if gt in pred:
                yes_no_acc = 1
            else:
                yes_no_acc = 0
        else:
            yes_no_acc = 0
        return {
            "yes/no accuracy": yes_no_acc
        }

    else:
        exact_score = calculate_exactmatch(pred, gt)
        f1_score, precision, recall = calculate_f1score(pred, gt)
        b_score = sentence_bleu(references=[str(gt).lower().split()],
                                hypothesis=str(pred).lower().split())
        b_score_1 = sentence_bleu(references=[str(gt).lower().split()],
                                  hypothesis=str(pred).lower().split(), weights=(1, 0, 0, 0))
        b_score_2 = sentence_bleu(references=[str(gt).lower().split()],
                                  hypothesis=str(pred).lower().split(), weights=(0, 1, 0, 0))
        b_score_3 = sentence_bleu(references=[str(gt).lower().split()],
                                  hypothesis=str(pred).lower().split(), weights=(0, 0, 1, 0))
        return {
            'exact match score': exact_score,
            'f1 score': f1_score,
            'precision': precision,
            'recall': recall,
            'bleu_score': b_score,
            'bleu_score_1': b_score_1,
            'bleu_score_2': b_score_2,
            'bleu_score_3': b_score_3
        }


def main():
    pred_df = pd.read_json("/nvme/VLMRobustness/test_results.json")
    results = []
    #iterate dataframe
    for _, row in pred_df.iterrows():
        pred = row['pred']
        gt = row['gt']
        pred = row['pred']
        answer_type = row['answer_type']
        metrics_dict = evaluate(gt=gt, pred=pred, answer_type=answer_type)
        results.append({
            "qid": row['qid'],
            "answer_type": answer_type,
            "metrics": metrics_dict,
        })

    with open("/nvme/VLMRobustness/metrics.json", 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    # TODO add args
    #args = parse_option()
    #main(args)
    main()