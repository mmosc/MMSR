import re
from typing import Callable, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_similarity(
    features_df: pd.DataFrame,
    sim_function: Callable[[np.array, np.array], np.array],
    batches: int = 100,
) -> pd.DataFrame:
    """
    :param features_df: full Data array
    :param sim_function: similarity function, receiving two 2 dimensional data matrices
    :param batches: split arr into chunks for less RAM usage
    :return: the full similarity matrix
    """
    features = features_df.to_numpy()
    splits = np.array_split(features, batches, axis=0)
    similarity = np.zeros((len(features),) * 2, dtype=np.float32)
    y = 0
    for b in tqdm(splits):
        similarity[:, y : y + b.shape[0]] = sim_function(features, b)
        y += b.shape[0]
    return pd.DataFrame(
        data=similarity, index=features_df.index, columns=features_df.index.values
    )


def compute_top_ids(
    similarity_df: pd.DataFrame, top: int = -1, batches: int = 100
) -> pd.DataFrame:
    """
    :param similarity_df: a similarity matrix
    :param top: how many ids should get retrieved
    :param batches: split arr into chunks for less RAM usage
    :return: the ranking matrix
    """
    similarity = similarity_df.to_numpy()
    if top < 0:
        top = len(similarity)
    splits = np.array_split(similarity, batches, axis=0)
    ids = np.zeros((len(similarity), top), dtype=np.int32)
    y = 0
    for b in tqdm(splits):
        ids[y : y + b.shape[0], :] = np.argsort(b * -1, axis=1)[:, :top]
        y += b.shape[0]
    return pd.DataFrame(
        data=ids, index=similarity_df.index, columns=similarity_df.index.values
    )


def compute_top_ids_directly(
    features_df: pd.DataFrame,
    sim_function: Callable[[np.array, np.array], np.array],
    batches: int = 100,
    top: int = -1,
) -> np.array:
    """
    Calculates the final ranking matrix, in batches, with minimum memory requirements
    :param features_df: Input features of shape (samples, features)
    :param sim_function: Similarity function returning similarity scores for two matrices. The sample count may differ based on batching.
    :param batches: Batching reduces the intermediate memory requirements and improves process response times
    :param top: How many results to store, default are all
    :return: The ranking matrix
    """
    features = features_df.to_numpy()

    splits = np.array_split(features, batches, axis=0)
    splits_idx = np.array_split(np.arange(features.shape[0]), batches, axis=0)

    if top < 0:
        top = features.shape[0]

    top_values = np.zeros((features.shape[0], top), dtype=np.int32)
    for b, i in tqdm(list(zip(splits, splits_idx))):
        size_batch = b.shape[0]

        # Calculate similarities
        results = sim_function(features, b).T

        # Set the distance to the same document to -1 because we don't want it at the start.
        results[(np.arange(size_batch), i)] = -1

        # Get the document indices instead of the distances
        top_values[i, :] = np.argsort(results * -1, axis=1)[:, :top]

    return pd.DataFrame(data=top_values, index=features_df.index)


def get_genres(field: str) -> List[str]:
    """
    Parses the genre list string as provided in the csv
    :param field: The genres as a joined string
    :return: The genres as a list
    """
    return re.findall(r"\'(.*?)\'", field)


def get_mapping(df: pd.DataFrame):
    """
    Since string indices are unhandy, a mapping to integer indices are preferred
    :param df: A dataframe containing all items in the index
    :return: an id-to-key and a key-to-id mapping
    """
    id_to_key = sorted(df.index.values)
    key_to_id = dict(zip(id_to_key, list(range(len(df.index.values)))))
    return id_to_key, key_to_id


def get_genre_matrix(genres: pd.DataFrame) -> pd.DataFrame:
    """Returns the song-genre matrix
    :param genres: The genre dataframe
    :return: A boolean dataframe of shape(samples, genres), where samples and genres are sorted
    """

    # This is to get a list of the available genres and also its frequency
    all_genres = set()
    for song in genres["genre"]:
        all_genres = all_genres.union(get_genres(song))

    genre_id_to_key = sorted(list(all_genres))
    genre_key_to_id = dict(zip(genre_id_to_key, list(range(len(all_genres)))))

    id_to_key, key_to_id = get_mapping(genres)
    genre_matrix = np.zeros((len(genres), len(all_genres)), dtype=np.float32)

    for sample_index, sample_id in enumerate(id_to_key):
        for g in get_genres(genres["genre"].loc[sample_id]):
            genre_matrix[sample_index, genre_key_to_id[g]] = 1.0

    return pd.DataFrame(genre_matrix, columns=genre_id_to_key, index=id_to_key)
