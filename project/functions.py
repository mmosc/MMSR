import re
import numpy as np
from tqdm import tqdm

def get_genres(field):
    return re.findall(r"\'(.*?)\'", field)

# Check if any genre of the song one is in the genres of song two, if yes returns True
def isResultRelevant(songOneGenres, songTwoGenres):
    return any(item in get_genres(songOneGenres) for item in get_genres(songTwoGenres))

# Improved  similarities calculation
def get_cosine_similarity(arr_a, arr_b):
    norms_a = np.linalg.norm(arr_a, axis=-1)[:, np.newaxis]
    norms_b = np.linalg.norm(arr_b, axis=-1)[:, np.newaxis]
    divisor = norms_a * norms_b.T
    dotp = arr_a @ arr_b.T
    r = np.divide(dotp, divisor, np.zeros_like(divisor), where=divisor > 0)
    return r

def get_jaccard_similarity(arr_a: np.array, arr_b: np.array):
    norms_a = np.linalg.norm(arr_a, axis=-1)[:, np.newaxis]
    norms_b = np.linalg.norm(arr_b, axis=-1)[:, np.newaxis]
    divisor = norms_a + norms_b.T
    dotp = arr_a @ arr_b.T
    r = dotp / (divisor-dotp)
    return r
def get_innerProduct_similarity(arr_a: np.array, arr_b: np.array):
    return arr_a @ arr_b.T
   

def getSongIdByQuery(artist, track):
    id_ = info[(info['artist'] == artist) & (info['song'] == track)]
    if len(id_) == 0: # If the data entered dont return any song
        return None
    return id_.index.values[0]


def compute_in_batches_distance(arr_a: np.array, arr_b: np.array, simfunction, batches:int = 1):
    splits_b = np.array_split(arr_b, batches, axis=0)
    r = []
    for b in tqdm(splits_b):
        r.append(simfunction(arr_a, b))
    return np.concatenate(r, axis=1)

def compute_in_batches_topIds(results: np.array, idx_values: np.array, top: int=100, batches:int = 1 ):
    splits_b = np.array_split(results, batches, axis=0)
    return np.concatenate([idx_values[np.argsort(b * -1, axis=1)][:,:top] for b in tqdm(splits_b)], axis=0)

def compute_topIds(results: np.array, idx_values: np.array, top: int=100):
    return np.array(idx_values[np.argsort(results * -1, axis=-1)[0,:top]])

# Check if any genre of the song one is in the genres of song two, if yes returns True
def isResultRelevant(songOneGenres, songTwoGenres):
    return any(item in get_genres(songOneGenres) for item in get_genres(songTwoGenres))

def meanAveragePrecision(dfTopIds, topNumber, genres):
    
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
            AP = (1/REL) * np.sum(np.multiply(relevant_results, np.divide(np.cumsum(relevant_results,axis=0), np.arange(1,topNumber+1))))
        AP_.append(AP)
    return np.mean(AP_)

def meanReciprocalRank(dfTopIds, topNumber, genres):
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


def ndcgMean(dfTopIds, topNumber, genres):
    ndcg = []

    for queryId in tqdm(dfTopIds.index.values):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        sorted_results = sorted(relevant_results, reverse=True) 
        dcg =[ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(relevant_results)]
        idcg =[ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(sorted_results)]
        
        if idcg == 0: # Case when there is no relevant result in the top@K
            ndcg.append(0)
        else:
            ndcg.append(np.sum(dcg) / np.sum(idcg))
    return (ndcg, np.mean(ndcg))

def getMetrics(dfTopIds, topNumber, genres):

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
        idcg = np.sum([ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(sorted_results)])
        if idcg == 0: # Case when there is no relevant result in the top@K
            ndcg.append(0)
        else:
            ndcg.append(dcg / idcg)
    return (np.mean(AP_), np.mean(RR), np.mean(ndcg))

