from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc_and_samples(
        x: np.array,
        yt: np.array,
        yp: np.array,
        fpr: List[np.array],
        tpr: List[np.array],
        mean_fpr: np.array,
        mean_tpr: np.array,
        mean_auc: float,
        highlight: int = -1):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs[0, 0].imshow(x)
    axs[0, 1].imshow(yt)
    axs[0, 2].imshow(yp)

    hx, wx = x.shape[:2]
    hy, wy = yt.shape[:2]
    h2 = (hy - hx) // 2
    w2 = (wy - wx) // 2
    axs[0, 0].set_xlim(-w2, wx + w2)
    axs[0, 0].set_ylim(hx + h2, -h2)

    for k in range(len(fpr)):
        if k == highlight:
            st = 'g--'
            alpha = 1
        else:
            st = 'b--'
            alpha = 0.25

        axs[0, 3].plot(fpr[k], tpr[k], st, alpha=alpha)

    axs[0, 3].plot(mean_fpr, mean_tpr, 'r', label='Mean ROC', lw=2)
    axs[0, 3].plot([0, 1], [0, 1], 'k--')

    for j, p in enumerate([.2, .4, .6, .8]):
        axs[1, j].imshow((yp > (p * 255)).astype(np.uint8) * 255)
        axs[1, j].set_title(f'Prediction $\\tau={p:.2}$')
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])

    # Set titles
    axs[0, 0].set_title('Input')
    axs[0, 1].set_title('Ground Truth')
    axs[0, 2].set_title('Prediction')
    axs[0, 3].set_title(f'ROC Mean AUC = {mean_auc:.2f}')
    axs[0, 3].set_xlabel('False Positive Rate')
    axs[0, 3].set_ylabel('True Positive Rate')

    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    fig.tight_layout()

    return fig, axs
