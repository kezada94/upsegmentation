import argparse

import torch


class FloatAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) not in [1, 3]:
            raise argparse.ArgumentTypeError("Must provide either one float or three floats.")
        setattr(namespace, self.dest, values)


def labels_to_one_hot(target):
    return torch.nn.functional.one_hot(target, -1).transpose(1, 4).squeeze(-1)


def one_hot_to_labels(target):
    return torch.unsqueeze(torch.argmax(target, dim=1), dim=1)
