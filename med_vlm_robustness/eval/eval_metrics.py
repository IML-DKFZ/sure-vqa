import math
from eval.utils import *
from eval.glossary import *
import json
from nltk.translate.bleu_score import sentence_bleu

def bleu(candidate, references, n, weights):
    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is bigger than n)"
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is smaller than n)"
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return str(bleu_result)


# BLEU
def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


# Exact match
def calculate_exactmatch(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return 0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


# Exact match with normalization
def similarity_candidate_prediction(candidate_answer, prediction):
    candidate_answer = split_sentence(candidate_answer, 1)

    count = 0
    total = 0
    for word in prediction:
        if word in candidate_answer:
            count += 1

    total = len(candidate_answer)

    if total == 0:
        return 0.0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


def argmax(lst):
    return lst.index(max(lst))


def calculate_appearance_with_normalization(prediction, reference, candidate_set):
    prediction = normalize_word(prediction)
    reference = normalize_word(reference)
    prediction_words = split_sentence(prediction, 1)
    reference_words = split_sentence(reference, 1)

    candidate_set = candidate_set['0']

    similarity_list = []
    candidate_answer_normalized_list = []
    for candidate_answer in candidate_set:

        if isinstance(candidate_answer, int):
            candidate_answer = str(candidate_answer)

        candidate_answer = normalize_word(candidate_answer)
        candidate_answer_normalized_list.append(candidate_answer)
        similarity_list.append(similarity_candidate_prediction(candidate_answer, prediction_words))

    final_prediction = candidate_answer_normalized_list[argmax(similarity_list)]

    # import pdb; pdb.set_trace()

    if final_prediction == reference:
        return 1.0  #
    else:
        return 0.0


# F1
def calculate_f1score(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0, 0, 0  # "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall
        
def get_accuracy(eval_data, data_categories):

    sum_correct = 0 
    sum_num_categories = 0
    num_qs = 0
    results = [{
            'avg_accuracy': 0,
            'avg_num_categories' : 0,
        },]
    for i,line in eval_data.iterrows():
        qid = line['qid']
        gt = line['gt'].lower()
        pred = line['pred'].lower()
        answer_type = line['answer_type']
        # get the numberof categories and list of categories from the data_category dataframe
        num_categories = data_categories[data_categories['qid'] == qid]['num_categories'].iloc[0]
        
        if answer_type == 'CLOSED': # calculate the metrics if the sample is closed ended
            sum_num_categories += num_categories
            num_qs +=  1 
            if ((gt in pred) or (pred in gt)) and ('yes, no' not in pred):
                sum_correct += 1
                line['accuracy'] = 1
            else:
                line['accuracy'] = 0
            results.append(line.to_dict())
    
    results[0]['avg_accuracy'] = sum_correct / num_qs
    results[0]['avg_num_categories'] = sum_num_categories / num_qs
                
    return results

def get_open_ended_metrics(gt, pred):
    gt = gt.lower()
    pred = pred.lower()

    gt = normalize_word(gt)
    pred = normalize_word(pred)

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
        'exact_match_score': exact_score,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'bleu_score': b_score,
        'bleu_score_1': b_score_1,
        'bleu_score_2': b_score_2,
        'bleu_score_3': b_score_3,
    }
