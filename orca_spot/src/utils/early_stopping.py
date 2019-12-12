"""
Module: early_stopping.py
Authors: Christian Bergler, Hendrik Schroeter
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 12.12.2019
"""

"""
Early stopping criterion as a regularization in order to stop the training after no changes with respect to a chosen
validation metric to avoid overfitting.

Code from https://github.com/pytorch/pytorch/pull/7661
Access Data: 12.09.2018, Last Access Date: 08.12.2019
"""
class EarlyStoppingCriterion(object):

    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0
        assert mode in {"min", "max"}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0
        self._best_score = None
        self.is_improved = None

    def step(self, cur_score):
        if self._best_score is None:
            self._best_score = cur_score
            return False
        else:
            if self.mode == "max":
                self.is_improved = cur_score >= self._best_score + self.min_delta
            else:
                self.is_improved = cur_score <= self._best_score - self.min_delta

            if self.is_improved:
                self._count = 0
                self._best_score = cur_score
            else:
                self._count += 1
            return self._count > self.patience

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
