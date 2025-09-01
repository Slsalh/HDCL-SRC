#!/usr/bin/env python
# coding:utf-8

import numpy as np

import time
from collections import defaultdict

def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, id2label, new_label_dict, r_hiera, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict.flatten())
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1
                    break

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    # Should between 0 and 1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'macro_precision': precision_macro,
            'macro_recall': recall_macro,
        'micro_precision': precision_micro,
            'micro_recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'right_total': right_total,
            'predict_total': predict_total,
            'gold_total': gold_total}

def evaluate_by_level(epoch_predicts, epoch_labels, id2label, new_label_dict, r_hiera, depths, threshold=0.5, top_k=None):
    # set non depth level to 0, remove from gold labels
    max_depths = np.max(np.array(depths))

    depth_scores = {}
    for i in range(1, min(max_depths, 5)):
        depth_pred = epoch_predicts.copy()
        depth_gold = epoch_labels.copy()
        depth_id2label = id2label.copy()
        mask = np.array(depths) == i
        depth_pred = np.array(depth_pred)[mask]

        depth_label = np.where(np.array(depths) == i)[0]
        for j in range(len(depth_gold)):
            depth_gold[i] = [label for label in depth_gold[i] if label in depth_label]
        
        # remove key from id2label if not in depth_label
        depth_id2label = {k: v for k, v in depth_id2label.items() if k in depth_label}

        depth_scores[str(i)] = evaluate(epoch_predicts=depth_pred, epoch_labels=depth_gold, id2label=depth_id2label, new_label_dict=new_label_dict, r_hiera=r_hiera, threshold=threshold, top_k=top_k, skip_consistency=True)

    if max_depths > 5:
        depth_pred = epoch_predicts.copy()
        depth_gold = epoch_labels.copy()
        depth_id2label = id2label.copy()
        mask = np.array(depths) >= 5
        depth_pred = np.array(depth_pred)[mask]

        depth_label = np.where(np.array(depths) >= 5)[0]
        for j in range(len(depth_gold)):
            depth_gold[i] = [label for label in depth_gold[i] if label in depth_label]
        
        # remove key from id2label if not in depth_label
        depth_id2label = {k: v for k, v in depth_id2label.items() if k in depth_label}

        depth_scores[">=5"] = evaluate(epoch_predicts=depth_pred, epoch_labels=depth_gold, id2label=depth_id2label, new_label_dict=new_label_dict, r_hiera=r_hiera, threshold=threshold, top_k=top_k, skip_consistency=True)
    return depth_scores

