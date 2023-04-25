import os
from typing import Callable

import numpy as np
from datatable import dt

from src.similarity_functions import get_jaccard_similarity, get_overlap_similarity
from src.utils import get_genre_matrix, compute_in_batches_distance


def get_genre_df():
    path = "./../task2/id_genres_mmsr.tsv"
    return dt.fread(path).to_pandas().set_index("id")


def get_relevance(
    key: str, sim_function: Callable[[np.array, np.array], np.array]
) -> np.array:
    path = "cache/" + key + ".npy"
    if os.path.exists(path):
        return np.load(path)
    else:
        # Get the genres encoded as a boolean matrix
        genre_matrix = get_genre_matrix(get_genre_df())

        # Get the similarity
        data = compute_in_batches_distance(
            genre_matrix.to_numpy(dtype=np.float32), sim_function, 100
        )

        # Cache and return
        np.save(path, data)
        return data


os.makedirs("cache", exist_ok=True)


def get_jaccard_relevance():
    return get_relevance("jaccard_genre_relevance", get_jaccard_similarity)


def get_simple_relevance():
    return get_relevance("simple_genre_relevance", get_overlap_similarity)
