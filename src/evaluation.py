from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


def evaluate_similarity(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    evaluation_function: Callable[[pd.DataFrame, pd.DataFrame], float],
    batches: int = 100,
) -> float:
    """
    Evaluates the predicted similarities on the target relevance
    :param y_true: Relevance matrix
    :param y_pred: Predicted similarity matrix
    :param evaluation_function: Evaluation function, e.g. sklearn metrics like NDCG score
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


def get_metrics(relevance: pd.DataFrame, top_ids: pd.DataFrame, k: int = -1) -> dict:
    """
    :param relevance: Relevance matrix
    :param top_ids: The predicted top ids
    :param k: Optionally the k to evaluate on
    :return: A dictionary containing all metrics
    """
    RR = []
    AP_ = []
    ndcg = []

    # todo on float relevance matrices RR and AP fails

    for index in tqdm(range(len(top_ids))):
        top_k_ids = top_ids.values[index, :k]

        # Relevance of fetched results
        result_relevance = relevance.values[index, top_k_ids]

        # Construct the optimal order and (sorted) optimal relevance
        optimal_top_ids = np.argsort(relevance.values[index, :] * -1)[:k]
        sorted_results = relevance.values[index, optimal_top_ids]

        # MAP
        REL = np.sum(result_relevance)
        if REL == 0:  # Case when there is no relevant result in the top@K
            AP = 0
        else:
            AP = (1 / REL) * np.sum(
                np.multiply(
                    result_relevance,
                    np.divide(np.cumsum(result_relevance, axis=0), np.arange(1, k + 1)),
                )
            )
        AP_.append(AP)

        # MRR
        if np.count_nonzero(result_relevance) > 0:
            min_idx_rel = np.argmax(result_relevance > 0) + 1
            RR.append(1 / min_idx_rel)
        else:  # Case when there is no relevant result in the top@K
            RR.append(0)

        # NDCG
        dcg = np.sum(
            [
                res / np.log2(i + 1) if i + 1 > 1 else float(res)
                for i, res in enumerate(result_relevance)
            ]
        )
        idcg = np.sum(
            [
                res / np.log2(i + 1) if i + 1 > 1 else float(res)
                for i, res in enumerate(sorted_results)
            ]
        )
        ndcg.append(0 if idcg == 0 else dcg / idcg)

    return {"MAP": np.mean(AP_), "MRR": np.mean(RR), "NDCG": np.mean(ndcg)}
