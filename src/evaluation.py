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


def get_metrics(top_ids: pd.DataFrame, relevance: pd.DataFrame, k: int = -1) -> dict:
    """
    :param relevance: Relevance matrix
    :param top_ids: The predicted top ids
    :param k: Optionally the k to evaluate on
    :return: A dictionary containing all metrics
    """
    RR = []
    AP_ = []
    ndcg = []

    if k < 0:
        k = relevance.shape[1]

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


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def custom_tau_kendall(r1, r2, K):
    possible_pairs_r1 = [(a, b) for idx, a in enumerate(r1) for b in r1[idx + 1 :]]
    possible_pairs_r2 = [(a, b) for idx, a in enumerate(r2) for b in r2[idx + 1 :]]
    concordant_pairs = intersection(possible_pairs_r1, possible_pairs_r2)
    # 2 times size of concordant pairs because they are repeated in the two rankings
    delta = (K * (K - 1)) - (len(concordant_pairs) * 2)
    tau = 1 - ((2 * delta) / (K * (K - 1)))
    return tau


def evaluate_ranking(
    y_true: pd.DataFrame, y_predicted: pd.DataFrame, correlation_measure: Callable
) -> float:
    score = 0
    samples = 0
    with tqdm(y_true.index.values) as t:
        for i in t:
            c = correlation_measure(y_true.loc[i], y_predicted.loc[i])
            score += c
            samples += 1
            t.set_description(f"Correlation: {score / samples}")
    return score / samples
