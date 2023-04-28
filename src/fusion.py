from typing import Callable, List

import numpy as np
import pandas as pd

from src.data import get_features
from src.pca import get_PCA_selection
from src.utils import compute_similarity, compute_top_ids, compute_top_ids_directly


def late_fusion(
    feature_names: List[str],
    sim_function: Callable,
    weights: List[float] = None,
    std_subsampling: int = 64,
) -> pd.DataFrame:
    """
    Returns the late fusion of given feature names using similarity summation
    :param feature_names: List of feature name in the datasets
    :param sim_function: Similar function to generate similarity matrices
    :param weights: Optional feature weights
    :param std_subsampling: Std calculation for normalization is slow and memory intensive, subsampling is recommended
    :return:
    """
    n = None

    for i, feature_name in enumerate(feature_names):
        result = compute_similarity(get_features(feature_name), sim_function).to_numpy()

        # Weight
        weight = 1 if weights is None else weights[i]

        # Normalize mean
        np.subtract(result, result.mean(), out=result)

        # Normalize std and apply weight
        result = np.multiply(
            result,
            weight / result[::std_subsampling, ::std_subsampling].std(),
            out=result,
        )

        # Sum
        if n is None:
            n = result
        else:
            np.add(n, result, n)

    return compute_top_ids(pd.DataFrame(n))


def early_fusion(
    feature_names: List[str],
    sim_function: Callable,
    weights: List[float] = None,
    std_subsampling: int = 64,
) -> pd.DataFrame:
    """
    Returns the early fusion of given feature names by performing dimension reduction and concatenation
    :param feature_names: List of feature name in the datasets
    :param sim_function: Similar function to generate similarity matrices
    :param weights: Optional feature weights
    :param std_subsampling: Std calculation for normalization is slow and memory intensive, subsampling is recommended
    :return:
    """
    df = None

    for i, feature_name in enumerate(feature_names):
        result_df = get_features(feature_name)
        result = get_PCA_selection(result_df).to_numpy()

        # Weight
        weight = 1 if weights is None else weights[i]

        # Normalize mean
        np.subtract(result, result.mean(), out=result)

        # Normalize std and apply weight
        result = np.multiply(
            result,
            weight / result[::std_subsampling, ::std_subsampling].std(),
            out=result,
        )

        # Sum
        result_df = pd.DataFrame(result, index=result_df.index)
        if df is None:
            df = result_df
        else:
            df.join(result_df, on="id", lsuffix="_")

    return compute_top_ids_directly(df, sim_function)
