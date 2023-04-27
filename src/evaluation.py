from typing import Callable

import numpy as np
from tqdm import tqdm

from src.utils import get_genres


def is_result_relevant(songOneGenres, songTwoGenres):
    """Check if any genre of the song one is in the genres of song two, if yes returns True"""
    return any(item in get_genres(songOneGenres) for item in get_genres(songTwoGenres))


def evaluate_similarity(
    y_true: np.array,
    y_pred: np.array,
    evaluation_function: Callable[[np.array, np.array], np.array],
    batches: int = 100,
) -> float:
    """
    Evaluates the predicted similarities on the target relevance
    :param y_true: Relevance matrix
    :param y_pred: Predicted similarity matrix
    :param evaluation_function: Evaluation function, e.g. sklearn metrics
    :param batches: Batches to process evaluation function
    :return: A float from 0 to 1 representing the score
    """
    y_true_splits = np.array_split(y_true, batches, axis=0)
    y_pred_splits = np.array_split(y_pred, batches, axis=0)
    score = 0
    samples = 0
    with tqdm(list(zip(y_true_splits, y_pred_splits))) as t:
        for (y_true_split, y_pred_split) in t:
            score += evaluation_function(y_true_split, y_pred_split) * len(y_true_split)
            samples += len(y_true_split)
            t.set_description(f"Score: {score / samples}")
    return score / samples


def get_metrics(top_id_df, top_k, genres):
    RR = []
    AP_ = []

    for queryId in tqdm(top_id_df.index.values):
        topIds = top_id_df.loc[queryId].values[:top_k]
        querySongGenres = genres.loc[[queryId], "genre"].values[0]
        topSongsGenres = genres.loc[topIds, "genre"].values
        relevant_results = [
            is_result_relevant(querySongGenres, songGenre)
            for songGenre in topSongsGenres
        ]

        # MAP
        REL = np.sum(relevant_results)
        if REL == 0:  # Case when there is no relevant result in the top@K
            AP = 0
        else:
            AP = (1 / REL) * np.sum(
                np.multiply(
                    relevant_results,
                    np.divide(
                        np.cumsum(relevant_results, axis=0), np.arange(1, top_k + 1)
                    ),
                )
            )
        AP_.append(AP)

        # MRR
        if True in relevant_results:
            min_idx_rel = relevant_results.index(True) + 1
            RR.append(1 / min_idx_rel)
        else:  # Case when there is no relevant result in the top@K
            RR.append(0)
    return np.mean(AP_), np.mean(RR)
