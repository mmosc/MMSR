import re
from typing import Callable, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_in_batches_distance(
    arr: np.array,
    sim_function: Callable[[np.array, np.array], np.array],
    batches: int = 1,
):
    """
    :param arr: full Data array
    :param sim_function: similarity function, receiving two 2 dimensional data matrices
    :param batches: split arr into chunks for less RAM usage
    :return: the full similarity matrix
    """
    splits = np.array_split(arr, batches, axis=0)
    r = np.zeros((len(arr),) * 2, dtype=np.float32)
    y = 0
    for b in tqdm(splits):
        r[:, y : y + b.shape[0]] = sim_function(arr, b)
        y += b.shape[0]
    return r


def compute_in_batches_top_ids(results: np.array, top: int = -1, batches: int = 1):
    """
    :param results: a similarity matrix
    :param top: how many ids should get retrieved
    :param batches: split arr into chunks for less RAM usage
    :return: the ranking matrix
    """
    if top < 0:
        top = len(results)
    splits = np.array_split(results, batches, axis=0)
    ids = np.zeros((len(results), top), dtype=np.int32)
    y = 0
    for b in tqdm(splits):
        ids[y : y + b.shape[0], :] = np.argsort(b * -1, axis=1)[:, :top]
        y += b.shape[0]
    return ids


def compute_top(
    data: np.array,
    sim_function: Callable[[np.array, np.array], np.array],
    batches: int = 1,
    top: int = -1,
) -> np.array:
    """
    Calculates the final ranking matrix, in batches, with minimum memory requirements
    :param data: Input features of shape (samples, features)
    :param sim_function: Similarity function returning similarity scores for two matrices. The sample count may differ based on batching.
    :param batches: Batching reduces the intermediate memory requirements and improves process response times
    :param top: How many results to store, default are all
    :return: The ranking matrix
    """
    splits = np.array_split(data, batches, axis=0)
    splits_idx = np.array_split(np.arange(data.shape[0]), batches, axis=0)

    if top < 0:
        top = data.shape[0]

    top_values = np.zeros((data.shape[0], top), dtype=np.int32)
    for b, i in tqdm(list(zip(splits, splits_idx))):
        size_batch = b.shape[0]

        # Calculate similarities
        results = sim_function(data, b).T

        # Set the distance to the same document to -1 because we don't want it at the start.
        results[(np.arange(size_batch), i)] = -1

        # Get the document indices instead of the distances
        top_values[i, :] = np.argsort(results * -1, axis=1)[:, :top]

    return top_values


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
    :return: A boolean dataframe of shape(samples, genres)
    """

    # This is to get a list of the available genres and also its frequency
    all_genres = set()
    for song in genres["genre"]:
        all_genres = all_genres.union(get_genres(song))

    genre_id_to_key = sorted(list(all_genres))
    genre_key_to_id = dict(zip(genre_id_to_key, list(range(len(all_genres)))))

    id_to_key, key_to_id = get_mapping(genres)
    genre_matrix = np.zeros((len(genres), len(all_genres)), dtype=np.int32)

    for genre_id in range(len(genres)):
        for g in get_genres(genres["genre"].loc[id_to_key[genre_id]]):
            genre_matrix[genre_id, genre_key_to_id[g]] = 1

    return pd.DataFrame(genre_matrix, columns=genre_id_to_key, index=id_to_key)
