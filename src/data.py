import os
from typing import Callable

import numpy as np
from datatable import dt

from src.similarity_functions import get_jaccard_similarity, get_overlap_similarity
from src.utils import get_genre_matrix, compute_in_batches_similarity


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
        data = compute_in_batches_similarity(
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


def get_features(feature: str):
    assert feature in features

    if feature == "blf_logfluc":
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
        data = (
            dt.fread(features[feature], header=True)
            .to_pandas()
            .set_index("id")
            .astype(np.float32)
        )

    # Reindex to make sure the underlying numpy array also conforms to the universal order
    data = data.reindex(sorted(data.index.values), copy=False)

    return data
