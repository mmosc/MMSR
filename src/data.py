import os
import pickle
from typing import Callable

import numpy as np
import pandas as pd
from datatable import dt

from src.utils import get_genre_matrix, compute_similarity, compute_top_ids_directly

DATA_SIZE = -1


def get_genre_df():
    path = "./../task2/id_genres_mmsr.tsv"
    return dt.fread(path).to_pandas().set_index("id")


USE_CACHE = True

if USE_CACHE:
    os.makedirs("cache", exist_ok=True)


def get_cached(
    key: str,
    feature: np.array,
    processor: Callable,
    sim_function: Callable[[np.array, np.array], np.array],
) -> np.array:
    """
    Processes data and caches it on disk
    :param key: Unique id for that data combination
    :param feature: The feature matrix (samples, features)
    :param processor: One of the compute functions (similarity, or top ids)
    :param sim_function: The similarity function
    :return:
    """
    path = "cache/" + key + ".p"
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    else:
        # Get the similarity
        data = processor(feature, sim_function, min(len(feature), 100))

        # Cache and return
        if USE_CACHE:
            with open(path, "wb") as file:
                pickle.dump(data, file)
        return data


# noinspection SpellCheckingInspection
features = {
    "tfidf": "./../task2/id_lyrics_tf-idf_mmsr.tsv",
    "word2vec": "./../task2/id_lyrics_word2vec_mmsr.tsv",
    "bert": "./../task2/id_lyrics_bert_mmsr.tsv",
    "mfcc_bow": "./../task2/id_mfcc_bow_mmsr.tsv",
    "mfcc_stats": "./../task2/id_mfcc_stats_mmsr.tsv",
    "essentia": "./../task2/id_essentia_mmsr.tsv",
    "blf_delta_spectral": "./../task2/id_blf_deltaspectral_mmsr.tsv",
    "blf_correlation": "./../task2/id_blf_correlation_mmsr.tsv",
    "blf_logfluc": "./../task2/id_blf_logfluc_mmsr.tsv",
    "blf_spectral": "./../task2/id_blf_spectral_mmsr.tsv",
    "blf_spectral_contrast": "./../task2/id_blf_spectralcontrast_mmsr.tsv",
    "blf_vardelta_spectral": "./../task2/id_blf_vardeltaspectral_mmsr.tsv",
    "incp": "./../task2/id_incp_mmsr.tsv",
    "vgg19": "./../task2/id_resnet_mmsr.tsv",
    "resnet": "./../task2/id_vgg19_mmsr.tsv",
}


def get_features(feature: str) -> pd.DataFrame:
    if feature == "genre_matrix":
        data = get_genre_matrix(get_genre_df())
    elif feature == "blf_logfluc":
        data = dt.fread(features[feature])

        # This is done because in the csv it has an extra column name,
        # so in case someone with the original dataset tries to run it, it fixes that error
        # It looks weird, but it is because first I am loading the data into datatable and then pass it to dataframe
        new_cols = ["id"]
        new_cols.extend(list(data.names[2:]))
        new_cols = tuple(new_cols)
        del data[:, -1]

        data.names = new_cols
        data = data.to_pandas()
        data.set_index("id", inplace=True)
        data = data.astype(np.float32)
    else:
        assert feature in features

        data = (
            dt.fread(features[feature], header=True)
            .to_pandas()
            .set_index("id")
            .astype(np.float32)
        )

    # Reindex to make sure the underlying numpy array also conforms to the universal order
    data = data.reindex(sorted(data.index.values), copy=False)

    data = data.iloc[:DATA_SIZE]

    return data


def get_similarity_for(
    feature: str,
    sim_function: Callable[[np.array, np.array], np.array],
) -> np.array:
    """
    Cached helper to retrieve the top ids for a given feature name and similarity function
    :param feature:
    :param sim_function:
    :return:
    """
    return get_cached(
        feature + "_" + sim_function.__name__ + "_" + str(DATA_SIZE) + "_similarity",
        get_features(feature),
        compute_similarity,
        sim_function,
    )


def get_top_ids_for(
    feature: str,
    sim_function: Callable[[np.array, np.array], np.array],
) -> np.array:
    """
    Cached helper to retrieve the top ids for a given feature name and similarity function
    :param feature:
    :param sim_function:
    :return:
    """
    return get_cached(
        feature + "_" + sim_function.__name__ + "_" + str(DATA_SIZE) + "_top_ids",
        get_features(feature),
        compute_top_ids_directly,
        sim_function,
    )


def get_random_top_ids(k: int = -1) -> pd.DataFrame:
    reference = get_features("tfidf")

    size = DATA_SIZE
    if size < 0:
        size = len(reference)
    if k < 0:
        k = size

    top_random_ids = np.empty((size, k), dtype=np.int32)
    np.random.seed(42)
    for i in range(size):
        top_random_ids[i] = np.random.choice(size, k, replace=False)

    return pd.DataFrame(top_random_ids, index=reference.index)
