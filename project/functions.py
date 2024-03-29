import re
import numpy as np
from tqdm import tqdm
from os.path import exists
import pandas as pd

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
    divisor = norms_a**2 + norms_b.T**2
    dotp = arr_a @ arr_b.T
    r = dotp / (divisor-dotp)
    return r
def get_innerProduct_similarity(arr_a: np.array, arr_b: np.array):
    return arr_a @ arr_b.T
   

def getSongIdByQuery(artist, track, info):
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
    return any(item in songOneGenres for item in songTwoGenres)

def meanAveragePrecision(dfTopIds, topNumber, genres, relevantNumbers):
    
    AP_ = []
    for queryId in tqdm(dfTopIds.index.values):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]

        # Which one is correct? Relevants in top k, or in all the query i.e the 68841

        # Total of relevant for the query
#         REL =relevantNumbers.loc[queryId].values

        # Relevants in the top k
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

def ndcgMean(dfTopIds, topNumber, genres, relevantNumbers):
    ndcg = []

    for queryId in tqdm(dfTopIds.index.values):
        total_relevant = relevantNumbers.loc[queryId].values
        topIds = dfTopIds.loc[queryId].values[:topNumber]

        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres  = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        
        if(total_relevant >= topNumber):
            sorted_results = np.ones(topNumber)
        else:
            sorted_results = np.concatenate((np.ones(total_relevant), np.zeros(topNumber-total_relevant)))

        dcg =[ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(relevant_results)]
        idcg =[ res/np.log2(i+1) if i+1 > 1 else float(res) for i,res in enumerate(sorted_results)]
        
        if idcg == 0: # Case when there is no relevant result in the top@K
            ndcg.append(0)
        else:
            ndcg.append(np.sum(dcg) / np.sum(idcg))
    return (ndcg, np.mean(ndcg))

def getMetrics(dfTopIds, topNumber, genres, relevantNumbers):

    RR = []
    AP_ = []
    ndcg = []

    for queryId in tqdm(dfTopIds.index.values):
        
        total_relevant = relevantNumbers.loc[queryId].values
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres  = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        
        if(total_relevant >= topNumber):
            sorted_results = np.ones(topNumber)
        else:
            sorted_results = np.concatenate((np.ones(total_relevant), np.zeros(topNumber-total_relevant)))

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

# Only for complete Dataset
"""
Execution example
 top = compute_in_batches_top(
        tf_idf.to_numpy(dtype=np.float32), 
        tf_idf.to_numpy(dtype=np.float32), 
        get_cosine_similarity, 
        tf_idf.index.values, 
        batches=100,   
        top=100)
"""
def compute_in_batches_top(arr_a: np.array, arr_b: np.array, simfunction, idx_values, batches:int = 1, top: int=100):
    
    splits_b = np.array_split(arr_b, batches, axis=0)
    splits_idx = np.array_split(np.arange(arr_b.shape[0]), batches, axis=0)
    
    top_values = []
    init_state = 0
    for b,i in tqdm(list(zip(splits_b, splits_idx))):
        size_batch = b.shape[0]
        result = simfunction(arr_a, b).T
        result[(np.arange(size_batch),i)] = -1
        top_values.append(idx_values[np.argsort(result * -1, axis=1)][:,:top])
        init_state += b.shape[0]
              
    return np.concatenate(top_values, axis=0)


def generate_MAP_MRR_NDCG_file(file, datasets, genres, relevantNumbers):
    if exists(file ):
        metrics_datasets = pd.read_csv(file, index_col=0)
    else:
      
        MAP_100 = np.zeros((len(datasets.items())))
        MRR_100 = np.zeros((len(datasets.items())))
        MeanNDCG_100 = np.zeros((len(datasets.items())))

        MAP_10 = np.zeros((len(datasets.items())))
        MRR_10 = np.zeros((len(datasets.items())))
        MeanNDCG_10 = np.zeros((len(datasets.items())))

        i = 0
        for value in datasets.values():
            MAP_100[i], MRR_100[i], MeanNDCG_100[i] = getMetrics(value, 100, genres, relevantNumbers)
            MAP_10[i], MRR_10[i], MeanNDCG_10[i] = getMetrics(value, 10, genres, relevantNumbers)
            i += 1 

        metrics_datasets =pd.DataFrame(
            np.column_stack((MAP_10,MAP_100, MRR_10,MRR_100,MeanNDCG_10,MeanNDCG_100)), 
            index=datasets.keys(), 
            columns=['MAP_10','MAP_100','MRR_10','MRR_100','Mean NDCG_10','Mean NDCG_100'])
        metrics_datasets.to_csv(file)
        
    return metrics_datasets

# The same as generate_MAP_MRR_NDCG_file, but with all the metrics
def get_metrics_file(file, datasets, genres,relevantNumbers,spotifyData):
    if exists(file ):
        metrics_datasets = pd.read_csv(file, index_col=0)
    else:
        
        MAP_100 = np.zeros((len(datasets.items())))
        MRR_100 = np.zeros((len(datasets.items())))
        MeanNDCG_100 = np.zeros((len(datasets.items())))
        MAP_10 = np.zeros((len(datasets.items())))
        MRR_10 = np.zeros((len(datasets.items())))
        MeanNDCG_10 = np.zeros((len(datasets.items())))
        
        bias_values = np.zeros((len(datasets.items())))
        hubness_10 = np.zeros((len(datasets.items())))
        hubness_100 = np.zeros((len(datasets.items())))
    
        i = 0
        for value in datasets.values():
            MAP_100[i], MRR_100[i], MeanNDCG_100[i] = getMetrics(value, 100, genres, relevantNumbers)
            MAP_10[i], MRR_10[i], MeanNDCG_10[i] = getMetrics(value, 10, genres, relevantNumbers)
            bias_values[i] = np.median(get_popularity_bias_metric(value, spotifyData, np.mean))
            hubness_10[i] = metric_hubness(value, 10)
            hubness_100[i] = metric_hubness(value, 100)
            i+=1
                  
        metrics_datasets =pd.DataFrame(
            np.column_stack((MAP_10,MAP_100, MRR_10,MRR_100,MeanNDCG_10,MeanNDCG_100,bias_values,hubness_10,hubness_100)), 
            index=datasets.keys(), 
            columns=['MAP_10','MAP_100','MRR_10','MRR_100','Mean NDCG_10','Mean NDCG_100','%DeltaMean','S10', 'S100'])
        metrics_datasets.to_csv(file)
        
    return metrics_datasets

def Precision(dfTopIds, topNumber, returnMeanOfValues, genres, relevantNumbers):
    
    precision = np.zeros((dfTopIds.shape[0], topNumber))
    recall = np.zeros((dfTopIds.shape[0], topNumber))
    precision_max = np.zeros((dfTopIds.shape[0], topNumber))
    
    for idx,queryId in tqdm(enumerate(dfTopIds.index.values)):
        
        topIds = dfTopIds.loc[queryId].values[:topNumber]
        querySongGenres = genres.loc[[queryId], 'genre'].values[0]
        topSongsGenres  = genres.loc[topIds, 'genre'].values
        relevant_results = [isResultRelevant(querySongGenres, songGenre) for songGenre in topSongsGenres]
        
        #Relevants in all set
        REL =relevantNumbers.loc[queryId].values

        if REL != 0: # Case when there is no relevant result in the top@K
            precision[idx] = np.divide(np.cumsum(relevant_results,axis=0), np.arange(1,topNumber+1))
            recall[idx] = np.divide(np.cumsum(relevant_results,axis=0), REL)
            precision_max[idx] = [ np.max(precision[idx,i:]) for i,val in enumerate(precision[idx])]

    if returnMeanOfValues:
        return np.mean(precision, axis=0), np.mean(recall, axis=0), np.mean(precision_max, axis=0)
    return precision, recall, precision_max

def get_precision_data(f_p, f_r, f_p_max, datasets, genres, relevant_by_id):
    if (exists(f_p) and  exists(f_r) and exists(f_p_max)):
        P = pd.read_csv(f_p, index_col=0).to_numpy()
        R = pd.read_csv(f_r, index_col=0).to_numpy()
        P_max = pd.read_csv(f_p_max, index_col=0).to_numpy()
    else:
        i = 0
        P = np.zeros((len(datasets.items()), 100))
        R = np.zeros((len(datasets.items()), 100))
        P_max = np.zeros((len(datasets.items()), 100))
        for value in datasets.values():
            P[i], R[i], P_max[i] = Precision(value, 100, True, genres, relevant_by_id)
            i += 1  

        pd.DataFrame(P, index=datasets.keys()).to_csv(f_p)
        pd.DataFrame(R, index=datasets.keys()).to_csv(f_r)
        pd.DataFrame(P_max, index=datasets.keys()).to_csv(f_p_max)
        
    return P,R,P_max


def metric_hubness(dataset, k):
    _, counts = np.unique(dataset.values[:,:k], return_counts=True)
    s = ((counts - np.mean(counts))**3)/(np.std(counts)**3)
    return np.mean(s)

def get_popularity_bias_metric(topIdsDataset, popularityDataset,  metric):
    bias_metric = []
    for query in topIdsDataset.index.values:
        recommended_documents = topIdsDataset.loc[query].values
        recommended_documents_popularity = popularityDataset.loc[recommended_documents].values
        mean_recommended_documents_popularity = metric(recommended_documents_popularity)
        # Only the popularity of the query? Is this correct?
        query_popularity = popularityDataset.loc[query].item()
        if query_popularity != 0:
            popularity_bias = (mean_recommended_documents_popularity - query_popularity) / query_popularity
        else:
            popularity_bias = 0
        bias_metric.append(popularity_bias)
        
    return np.array(bias_metric)

# Functions for correlation metrics

def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def tauKendall(r1,r2, K):
    possible_pairs_r1 = [(a, b) for idx, a in enumerate(r1) for b in r1[idx + 1:]]
    possible_pairs_r2 = [(a, b) for idx, a in enumerate(r2) for b in r2[idx + 1:]]
    concordant_pairs = intersection(possible_pairs_r1, possible_pairs_r2)
    # 2 times size of concordant pairs because they are repeated in the two rankings
    delta = (K*(K-1)) - (len(concordant_pairs)*2)
    tau = 1 - ((2*delta) / (K*(K-1)))
    return tau

def getCorrelationTau(file_correlations, datasets):
    if exists(file_correlations):
        return pd.read_csv(file_correlations, index_col=0)
    else:    
        correlations_tau = pd.DataFrame(index=datasets.keys(), columns=datasets.keys())
        queries = datasets[list(datasets.keys())[0]].index.values
        df_list = list(datasets.keys())
        df_combinations_ = [(a, b) for idx, a in enumerate(df_list) for b in df_list[idx + 1:]]
        df_combinations =list(zip(df_list, df_list))
        df_combinations.extend(df_combinations_)
        corr_all_values_tau  = np.zeros((len(df_combinations), queries.shape[0]))
        print("Total iterations:", len(df_combinations))

        for i, (df1, df2) in tqdm(enumerate(df_combinations)):
            for idx,query in enumerate(queries):
                corr_all_values_tau[i, idx] = tauKendall(datasets[df1].loc[query].values, datasets[df2].loc[query].values, 100)
            correlations_tau.loc[df1, df2] = np.mean(corr_all_values_tau[i])

        correlations_tau.to_csv(file_correlations)
        return correlations_tau


# Helper function to transfrom between standard ids and new ids and viceversa
def from_standard_ids_to_new(df, relationIds):
    id_to_key = dict(zip(relationIds.index.values, relationIds['newId'].values))
    new_df = df.apply(lambda s : [id_to_key[x] for x in s] , raw=True , axis=1)
    new_df = new_df.set_index(np.asarray([id_to_key[i] for i in df.index.values]))
    new_df.index.astype(np.int32, copy=False)
    new_df.index.name='index'
    return new_df

def from_new_ids_to_standard(df, relationIds):
    id_to_key = dict(zip(relationIds['newId'].values, relationIds.index.values))
    new_df =  df.apply(lambda s : [id_to_key[x] for x in s] , raw=True , axis=1)
    new_df = new_df.set_index(np.asarray([id_to_key[i] for i in df.index.values]))
    new_df.index.name='index'
    return new_df