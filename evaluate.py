#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""evaluate script"""

from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import argparse
from io import open

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def load_labels(path):
    """Load labels from `path`"""
    pid_to_lbl = {}
    with open(path, 'rt', encoding='utf-8') as fin:
        for line in fin:
            pid, lbl = line.strip().split(',')
            pid_to_lbl[pid] = int(lbl)
    return pid_to_lbl

def main():
    """entry point"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('truth_path', help='path to the truth')
    parser.add_argument('pred_path', help='path to the predictions')
    args = parser.parse_args()

    truth = load_labels(args.truth_path)
    pred = load_labels(args.pred_path)

    if len(truth) != len(pred):
        raise RuntimeError(
            'The number of prediction records ({}) is not consistent with '
            'the number of truth records ({}).'
            .format(len(pred), len(truth)))

    truth_labels, pred_labels = [], []
    for pid in truth:
        if pid not in pred:
            raise RuntimeError(
                'Record `{}` is not in the predictions.'.format(pid))
        truth_labels.append(truth[pid])
        pred_labels.append(pred[pid])
    truth_labels = np.array(truth_labels)
    pred_labels = np.array(pred_labels)
    metrics = {
        'accuracy': accuracy_score(truth_labels, pred_labels),
        'macro_f1': f1_score(truth_labels, pred_labels, average='macro')
    }
    print(metrics)

if __name__ == '__main__':
    main()
