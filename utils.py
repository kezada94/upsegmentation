import numpy as np


def parallel_average(n_a, avg_a, m2_a, n_b, avg_b, m2_b):
    n_ab = n_a + n_b
    delta = avg_a - avg_b
    avg_ab = (avg_a * n_a + avg_b * n_b) / n_ab
    m2_ab = m2_a + m2_b + delta ** 2 * n_a * n_b / n_ab
    return n_ab, avg_ab, m2_ab


def get_mean_std_color(path, dataset):
    data = [
        (
            x.shape[0] * x.shape[1],
            np.mean(x, axis=(0, 1)),
            np.sum((x - np.mean(x, axis=(0, 1))) ** 2, axis=(0, 1)),
        )
        for x, _ in dataset
    ]
    n, avg, m2 = data[0]
    for n_i, m_i, m2_i in data[1:]:
        n, avg, m2 = parallel_average(n, avg, m2, n_i, m_i, m2_i)
    return avg, np.sqrt(m2 / (n - 1))
