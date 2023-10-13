import torch


def _reduction(x, reduction):
    if reduction is None:
        return x
    elif reduction == 'sum':
        return torch.sum(x)
    elif reduction == 'mean':
        return torch.mean(x)
    else:
        raise ValueError


def count_true_positives(yp, yt, threshold=0.5, reduction='sum'):
    c = ((yp > threshold) & (yt == 1)).sum(dim=(2, 3))
    return _reduction(c, reduction)


def count_true_negatives(yp, yt, threshold=0.5, reduction='sum'):
    c = ((yp < threshold) & (yt == 0)).sum(dim=(2, 3))
    return _reduction(c, reduction)


def count_false_positives(yp, yt, threshold=0.5, reduction='sum'):
    c = ((yp > threshold) & (yt == 0)).sum(dim=(2, 3))
    return _reduction(c, reduction)


def count_false_negatives(yp, yt, threshold=0.5, reduction='sum'):
    c = ((yp < threshold) & (yt == 1)).sum(dim=(2, 3))
    return _reduction(c, reduction)
