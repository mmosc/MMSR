# Fastapi imports
from enum import Enum
from typing import Union
from fastapi import FastAPI

# Imports for project
import pandas as pd
import numpy as np
from os.path import exists
import re
from tqdm import tqdm

# DATA FILES PROVIDED
file_tfidf = "./../task2/id_lyrics_tf-idf_mmsr.tsv"
file_word2vec = "./../task2/id_lyrics_word2vec_mmsr.tsv"
file_bert = "./../task2/id_lyrics_bert_mmsr.tsv"
file_genres = "./../task2/id_genres_mmsr.tsv"
file_info = "./../task2/id_information_mmsr.tsv"

# csv_topIdsFiles = {
#     "cosineSim_tfidf" : './topIds/top_ids_cosine_tfidf.csv',
#     "innerProduct_tfidf" : './topIds/top_ids_innerProduct_tfidf.csv',
#     "jaccardSim_tfidf" : './topIds/top_ids_jaccard_tfidf.csv',
#     "cosineSim_word2vec" : './topIds/top_ids_cosine_word2vec.csv',
#     "innerProduct_word2vec" : './topIds/top_ids_innerProduct_word2vec.csv',
#     "jaccardSim_word2vec" : './topIds/top_ids_jaccard_word2vec.csv',
#     "cosineSim_bert" : './topIds/top_ids_cosine_bert.csv',
#     "innerProduct_bert" : './topIds/top_ids_innerProduct_bert.csv',
#     "jaccardSim_bert" : './topIds/top_ids_jaccard_bert.csv',
# }

## Complete files
csv_topIdsFiles = {
    "cosineSim_tfidf" : './topIds/top_ids_cosine_tfidf_complete.csv',
    "innerProduct_tfidf" : './topIds/top_ids_innerProduct_tfidf_complete.csv',
    "jaccardSim_tfidf" : './topIds/top_ids_jaccard_tfidf_complete.csv',
    "cosineSim_word2vec" : './topIds/top_ids_cosine_word2vec_complete.csv',
    "innerProduct_word2vec" : './topIds/top_ids_innerProduct_word2vec_complete.csv',
    "jaccardSim_word2vec" : './topIds/top_ids_jaccard_word2vec_complete.csv',
    "cosineSim_bert" : './topIds/top_ids_cosine_bert_complete.csv',
    "innerProduct_bert" : './topIds/top_ids_innerProduct_bert_complete.csv',
    "jaccardSim_bert" : './topIds/top_ids_jaccard_bert_complete.csv',
}

# Improved  similarities calculation
def get_cosine_similarity(arr_a: np.array, arr_b: np.array):
#     def func(d1, d2, divisor):
#         return np.divide(d1 @ d2, divisor, np.zeros_like(divisor), where=divisor > 0)

#     norms_a = np.linalg.norm(arr_a, axis=-1)
#     norms_b = np.linalg.norm(arr_b, axis=-1) # todo why doesn't norms[indicies_test] work here?

#     r = np.zeros((len(arr_a), len(arr_b)))
#     for index, sample in enumerate(tqdm(arr_b)):
#         #print(index)
#         r[:, index] = func(arr_a, sample, norms_a * norms_b[index])
#     return r

    # Richi: this is faster since we do not need a loop
    norms_a = np.linalg.norm(arr_a, axis=-1)[:, np.newaxis]
    norms_b = np.linalg.norm(arr_b, axis=-1)[:, np.newaxis]
    divisor = norms_a * norms_b.T
    dotp = arr_a @ arr_b.T
    r = np.divide(dotp, divisor, np.zeros_like(divisor), where=divisor > 0)
    return r


def get_jaccard_similarity(arr_a: np.array, arr_b: np.array):
    def func(d1, d2, divisor):
        d1_d2_product = d1 @ d2
        return np.divide(d1_d2_product, divisor - d1_d2_product, np.zeros_like(divisor), where=divisor > 0)

    norms_a = np.linalg.norm(arr_a, axis=-1)
    norms_b = np.linalg.norm(arr_b, axis=-1) # todo why doesn't norms[indicies_test] work here?

    r = np.zeros((len(arr_a), len(arr_b)))
    for index, sample in enumerate(tqdm(arr_b)):
        r[:, index] = func(arr_a, sample, norms_a + norms_b[index])

    return r
def get_innerProduct_similarity(arr_a: np.array, arr_b: np.array):
    r = np.zeros((len(arr_a), len(arr_b)))
    for index, sample in enumerate(tqdm(arr_b)):
        r[:, index] = arr_a @ sample 
    return r  
   

def getSongIdByQuery(artist, track):
    id_ = info[(info['artist'] == artist) & (info['song'] == track)]
    if len(id_) == 0: # If the data entered dont return any song
        return None
    return id_.index.values[0]

def distanceToSongsImproved(idSong, similarity_function, df, features_vector):
    if idSong in df.columns.values:
        print("Already in data")
    else:
        print("Calculating distances")
        features_vector.loc[idSong].values
        distances = similarity_function(features_vector.to_numpy(), features_vector.loc[idSong].values.reshape(-1,1).T).T[0]
        df[idSong]  = distances   

def getTopValues(idSong, df_metricUsed):
    top_values = df_metricUsed[idSong].sort_values(ascending=False)
    return genres.loc[top_values.index].join(info, on="id", how="left")

# Helpers for evaluation
def get_genres(field):
    return re.findall(r"\'(.*?)\'", field)

# Check if any genre of the song one is in the genres of song two, if yes returns True
def isResultRelevant(songOneGenres, songTwoGenres):
    return any(item in get_genres(songOneGenres) for item in get_genres(songTwoGenres))

def meanAveragePrecision(dfTopIds, topNumber):
    
    AP_ = []
    for queryId in tqdm(dfTopIds.index.values):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        REL = np.sum(relevant_results)
        if REL == 0: # Case when there is no relevant result in the top@K
            AP = 0
        else:
#             AP = (1/REL) * np.sum([relevant_results[i] * (np.sum(relevant_results[:i+1]) / (i+1)) for i in range(topNumber)])
            AP = (1/REL) * np.sum(np.multiply(relevant_results, np.divide(np.cumsum(relevant_results,axis=0), np.arange(1,topNumber+1))))
        
        AP_.append(AP)
        
    return np.mean(AP_)

def meanReciprocalRank(dfTopIds, topNumber):
    RR = []
    for queryId in tqdm(dfTopIds.index.values):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]

        if True in relevant_results:
            min_idx_rel = relevant_results.index(True) + 1
            RR.append(1/min_idx_rel)
        else: # Case when there is no relevant result in the top@K
            RR.append(0)

       
    return np.mean(RR)

# Gain for the user is considered with the genre, if the song retrieved contains the genre the gain will be 1, 
# if not 0.

# Also the gain could be considered in descending order from k,..., 0

# For example:
# Given the array of results marked as relevant  [1, 0, 0, 1, 1] for @k = @5
# For the first consideration the user gain will be the same d1(1), d2(0), d3(0), d4(1), d5(1)

# Not implemented
# For the second one the user gain could be d1(5), d2(0), d3(0), d4(4), d5(3),
# reducing in one the relevance everytime a new relevant doc appears

def ndcgMean(dfTopIds, topNumber):
    ndcg = []

    for queryId in tqdm(dfTopIds.index.values):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        sorted_results = sorted(relevant_results, reverse=True)

        dcg = np.sum([ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(relevant_results)])
        idcg = np.sum([ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(sorted_results)])
        
        if idcg == 0: # Case when there is no relevant result in the top@K
            ndcg.append(0)
        else:
            ndcg.append(dcg / idcg)
#         print(dcg, idcg)
    return (ndcg, np.mean(ndcg))

def getMetrics(dfTopIds, topNumber):

    RR = []
    AP_ = []
    ndcg = []

    for queryId in tqdm(dfTopIds.index.values):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        sorted_results = sorted(relevant_results, reverse=True)

        # MAP
        REL = np.sum(relevant_results)
        if REL == 0: # Case when there is no relevant result in the top@K
            AP = 0
        else:
            AP = (1/REL) * np.sum(np.multiply(relevant_results, np.divide(np.cumsum(relevant_results,axis=0), np.arange(1,topNumber+1))))
        AP_.append(AP)

        # MRR
        if True in relevant_results:
            min_idx_rel = relevant_results.index(True) + 1
            RR.append(1/min_idx_rel)
        else: # Case when there is no relevant result in the top@K
            RR.append(0)

        # NDCG
        dcg = np.sum([ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(relevant_results)])
        # print([ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(relevant_results)])
        idcg = np.sum([ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(sorted_results)])
        if idcg == 0: # Case when there is no relevant result in the top@K
            ndcg.append(0)
        else:
            ndcg.append(dcg / idcg)
    return (np.mean(AP_), np.mean(RR), np.mean(ndcg))


# Dataframes with the data provided
tf_idf = pd.read_table(file_tfidf, index_col="id")
word2vec = pd.read_table(file_word2vec, index_col='id')
bert = pd.read_table(file_bert, index_col='id')
genres = pd.read_table(file_genres, index_col="id")
info = pd.read_table(file_info, index_col="id")



# The index depends in the features vector, so it is better to assign 
# it depending on which feature vector we are using
# change to bert.index or word2vec.index

# Data frames to store temp the distances
df_cosineDistance_tfidf = pd.DataFrame(index=tf_idf.index)
df_innerProductDistance_tfidf = pd.DataFrame(index=tf_idf.index)
df_jaccardDistance_tfidf = pd.DataFrame(index=tf_idf.index)

df_cosineDistance_word2vec = pd.DataFrame(index=word2vec.index) 
df_innerProductDistance_word2vec = pd.DataFrame(index=word2vec.index)  
df_jaccardDistance_word2vec = pd.DataFrame(index=word2vec.index)

df_cosineDistance_bert = pd.DataFrame(index=bert.index)    
df_innerProductDistance_bert = pd.DataFrame(index=bert.index)
df_jaccardDistance_bert = pd.DataFrame(index=bert.index)


# Can't store panda dataframes in an enum directly
dataVectorsMapping = {
    "tfidf" : tf_idf,
    "bert"  : bert,
    "word2vec" : word2vec
}

class DataVector(Enum):
    tfidf = "tf_idf"
    bert = "bert"
    word2vec = "word2vec"

    def to_df(self):
        return dataVectorsMapping[self.name]


similarityFunctionMapping = {
    "cosineSim" : get_cosine_similarity,
    "innerProduct" : get_innerProduct_similarity,
    "jaccardSim" : get_jaccard_similarity
}

class SimilarityFunction(Enum):
    cosineSim = "cosineSim"  # "Cosine Similarity"
    innerProduct = "innerProduct"  # "Inner Product Similarity"
    jaccardSim = "jaccardSim"  # "Jaccard Similarity"

    def to_func(self):
        return similarityFunctionMapping[self.name]

queryDistances = {
    "cosineSim_tfidf" : df_cosineDistance_tfidf,
    "innerProduct_tfidf" : df_innerProductDistance_tfidf,
    "jaccardSim_tfidf" : df_jaccardDistance_tfidf,
    "cosineSim_word2vec" : df_cosineDistance_word2vec,
    "innerProduct_word2vec" : df_innerProductDistance_word2vec,
    "jaccardSim_word2vec" : df_jaccardDistance_word2vec,
    "cosineSim_bert" : df_cosineDistance_bert,
    "innerProduct_bert" : df_innerProductDistance_bert,
    "jaccardSim_bert" : df_jaccardDistance_bert,
}

if exists(csv_topIdsFiles["cosineSim_tfidf"]):
    top_cosine_tfidf = pd.read_csv(csv_topIdsFiles["cosineSim_tfidf"] ,index_col=0)
else:
    top_cosine_tfidf = pd.DataFrame( columns=range(100))

if exists(csv_topIdsFiles["innerProduct_tfidf"]):
    top_innerProduct_tfidf = pd.read_csv(csv_topIdsFiles["innerProduct_tfidf"] ,index_col=0)
else:
    top_innerProduct_tfidf = pd.DataFrame( columns=range(100))

if exists(csv_topIdsFiles["jaccardSim_tfidf"]):
    top_jaccard_tfidf = pd.read_csv(csv_topIdsFiles["jaccardSim_tfidf"] ,index_col=0)
else:
    top_jaccard_tfidf = pd.DataFrame( columns=range(100))


if exists(csv_topIdsFiles["cosineSim_word2vec"]):
    top_cosine_word2vec = pd.read_csv(csv_topIdsFiles["cosineSim_word2vec"] ,index_col=0)
else:
    top_cosine_word2vec = pd.DataFrame( columns=range(100))

if exists(csv_topIdsFiles["innerProduct_word2vec"]):
    top_innerProduct_word2vec = pd.read_csv(csv_topIdsFiles["innerProduct_word2vec"] ,index_col=0)
else:
    top_innerProduct_word2vec = pd.DataFrame( columns=range(100))

if exists(csv_topIdsFiles["jaccardSim_word2vec"]):
    top_jaccard_word2vec = pd.read_csv(csv_topIdsFiles["jaccardSim_word2vec"] ,index_col=0)
else:
    top_jaccard_word2vec = pd.DataFrame( columns=range(100))


if exists(csv_topIdsFiles["cosineSim_bert"]):
    top_cosine_bert = pd.read_csv(csv_topIdsFiles["cosineSim_bert"] ,index_col=0)
else:
    top_cosine_bert = pd.DataFrame( columns=range(100))

if exists(csv_topIdsFiles["innerProduct_bert"] ):
    top_innerProduct_bert = pd.read_csv(csv_topIdsFiles["innerProduct_bert"]  ,index_col=0)
else:
    top_innerProduct_bert = pd.DataFrame( columns=range(100))

if exists(csv_topIdsFiles["jaccardSim_bert"]):
    top_jaccard_bert = pd.read_csv(csv_topIdsFiles["jaccardSim_bert"] ,index_col=0)
else:
    top_jaccard_bert = pd.DataFrame( columns=range(100))

topIdsFiles = {
    "cosineSim_tfidf" : top_cosine_tfidf,
    "innerProduct_tfidf" : top_innerProduct_tfidf,
    "jaccardSim_tfidf" : top_jaccard_tfidf,
    "cosineSim_word2vec" : top_cosine_word2vec,
    "innerProduct_word2vec" : top_innerProduct_word2vec,
    "jaccardSim_word2vec" : top_jaccard_word2vec,
    "cosineSim_bert" : top_cosine_bert,
    "innerProduct_bert" : top_innerProduct_bert,
    "jaccardSim_bert" : top_jaccard_bert,
}


app = FastAPI()

@app.get("/query/")
async def getTopResults(artist: str, track: str, top: int, vectorData: DataVector, simFunction: SimilarityFunction):
    print("Get Top results for \n\tArtist: ", artist, " \n\tTrack:",track, "\nUsing \n\tvectorData: ", vectorData.name, "\n\tsimilarity function: ", simFunction.name)

    id_song = getSongIdByQuery(artist, track)
    if id_song == None:
        return { "error" : "No record found for this query"}

    print("\nId song:", id_song)
    # print(top_jaccard_word2vec.head())

    file_id = simFunction.name + "_" + vectorData.name

    if id_song in  topIdsFiles[file_id].index.values:
        # topValues = getTopIds(id_song,  topIdsFiles[file_id])
        print('\nQuery already in Top ids file')
    else:
        print('\nNew song, calculating top 100 similar songs and saving to data')
        # distanceToSongs(id_song, simFunction , queryDistances[file_id], vectorData)
        distanceToSongsImproved(id_song, simFunction.to_func(), queryDistances[file_id], vectorData.to_df())
        print("Actual number of records:", len(topIdsFiles[file_id].index))
        # Add new record to top id file
        topIdsFiles[file_id].loc[id_song] = queryDistances[file_id][[id_song]].drop(axis=0, index=[id_song]).sort_values(by=id_song,ascending=False).head(100).index.values
        print("New number of records:", len(topIdsFiles[file_id].index))
        print("Writting to:", csv_topIdsFiles[file_id])
        # Update top id file in csv
        topIdsFiles[file_id].to_csv(csv_topIdsFiles[file_id])

    query_song = genres.loc[[id_song]].join(info, on="id", how="left")
    topVal = genres.loc[ topIdsFiles[file_id].loc[id_song].values].join(info, on="id", how="left").head(top)

    return { "song": query_song , "top": topVal }

@app.get("/metrics/")
async def getEvaluationMetrics(vectorData: DataVector, simFunction: SimilarityFunction, k:int):
    print("Get metrics for vectorData:", vectorData, "Using similarity funtion", simFunction.name)

    file_id = simFunction.name + "_" + vectorData.name

    pk, mrrk, ndcgk = getMetrics(topIdsFiles[file_id], k)
    # MAP
    # pk = meanAveragePrecision( topIdsFiles[file_id], k)
    print("MAP@"+str(k), pk)

    # # MRR
    # mrrk = meanReciprocalRank(topIdsFiles[file_id], k)
    print("MRR@"+str(k), mrrk)

    # # NDCG
    # _, ndcgk = ndcgMean(topIdsFiles[file_id], k)
    print("Mean NDCG@"+str(k), ndcgk)

    return { "SIM": simFunction.to_func(), "vectorData": vectorData.to_df(), "MAP" : pk, "MRR" : mrrk, "NDCG" : ndcgk }