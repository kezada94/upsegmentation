from typing import List

import torch


class Evaluations:
    def __init__(self, evals: List):
        self.evals = evals
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def __call__(self, yt, yp):
        self._true_positives += torch.sum((yp == 1) & (yt == 1)).item()
        self._true_negatives += torch.sum((yp == 0) & (yt == 0)).item()
        self._false_positives += torch.sum((yp == 1) & (yt == 0)).item()
        self._false_negatives += torch.sum((yp == 0) & (yt == 1)).item()

        return {ev: getattr(self, ev)() for ev in self.evals}

    def reset(self):
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def accuracy(self):
        """
            Accuracy is a measure of the overall correctness of the model, calculated as the ratio of correctly predicted
            instances to the total instances.

            Accuracy = Number of Correct Predictions / Total Number of Predictions

            :return: Accuracy score.
            """
        num_correct_predictions = self._true_positives + self._true_negatives
        num_predictions = self._true_positives + self._true_negatives + self._false_positives + self._false_negatives
        return num_correct_predictions / (num_predictions + 1e-8)

    def precision(self):
        """
        Precision measures the accuracy of the positive predictions. It is the ratio of correctly predicted positive
        observations to the total predicted positives.

        Precision = True Positives / (True Positives + False Positives)

        :return: Precision score.
        """
        return self._true_positives / (self._true_positives + self._false_positives + 1e-8)

    def recall(self):
        """
        Recall (Sensitivity or True Positive Rate) measures the ability of the model to capture all the relevant instances.
        It is the ratio of correctly predicted positive observations to all the actual positives.

        Recall = True Positives / (True Positives + False Negatives)

        :return: Recall score.
        """

        return self._true_positives / (self._true_positives + self._false_negatives + 1e-8)

    def f1_score(self):
        """
        The F1 Score is the harmonic mean of precision and recall. It provides a balanced measure between precision and
        recall.

        F1 Score = 2 * true positives / (2 * true positives + (false positives + false negatives))

        :return: F1 score.
        """
        return 2 * self._true_positives / (2 * self._true_positives + (self._false_positives + self._false_negatives) + 1e-8)


def auc(yp, yt):
    """
    AUC represents the area under the Receiver Operating Characteristic (ROC) curve. It is a measure of the model's
    ability to distinguish between positive and negative instances.
    Note: AUC values range from 0 to 1, where higher values indicate better performance.

    :param yp: Predicted labels or values.
    :param yt: True labels or values.
    :return: AUC score.
    """
    pass


def jaccard_index(yp, yt):
    """
    Jaccard Index measures the similarity between two sets by dividing the size of the intersection by the size of the
    union.

    Jaccard Index = Intersection of Predicted and True Sets / Union of Predicted and True Sets

    :param yp: Predicted labels or values.
    :param yt: True labels or values.
    :return: Jaccard Index score.
    """
    pass
