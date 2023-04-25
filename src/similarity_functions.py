import numpy as np


def get_cosine_similarity(arr_a: np.array, arr_b: np.array) -> np.array:
    norms_a = np.linalg.norm(arr_a, axis=-1)[:, np.newaxis]
    norms_b = np.linalg.norm(arr_b, axis=-1)[:, np.newaxis]
    divisor = norms_a * norms_b.T
    dot_p = arr_a @ arr_b.T
    return np.divide(dot_p, divisor, dot_p, where=divisor > 0)


def get_jaccard_similarity(arr_a: np.array, arr_b: np.array) -> np.array:
    norms_a = np.linalg.norm(arr_a, axis=-1)[:, np.newaxis]
    norms_b = np.linalg.norm(arr_b, axis=-1)[:, np.newaxis]
    divisor = norms_a ** 2 + norms_b.T ** 2
    dot_p = arr_a @ arr_b.T
    return dot_p / (divisor - dot_p)


def get_inner_product_similarity(arr_a: np.array, arr_b: np.array) -> np.array:
    return arr_a @ arr_b.T


def get_overlap_similarity(arr_a: np.array, arr_b: np.array) -> np.array:
    return arr_a @ arr_b.T > 0
