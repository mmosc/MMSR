# Fastapi imports
from enum import Enum
from fastapi import FastAPI

# Imports for project
import pandas as pd
from os.path import exists
import datatable as dt
from files import *
from functions import *


# Dataframes with the data provided
print("Loading Data....")
# To load all the dataset requires 15 GB of RAM approx. 
# To test in a system with less memory, reduce the size to float32, float16
# Or only load the datasets to test
tf_idf  = dt.fread(file_tfidf_2).to_pandas()
tf_idf.set_index('id', inplace=True)

word2vec  = dt.fread(file_word2vec_2, header=True).to_pandas()
word2vec.set_index('id', inplace=True)

bert = dt.fread(file_bert_2,header=True).to_pandas()
bert.set_index('id', inplace=True)

genres  = dt.fread(file_genres_2).to_pandas()
genres.set_index('id', inplace=True)

info  = dt.fread(file_info_2).to_pandas()
info.set_index('id', inplace=True)

blf_correlation  = dt.fread(file_blf_correlation).to_pandas()
blf_correlation.set_index('id', inplace=True)

blf_deltaspectral  = dt.fread(file_blf_deltaspectral).to_pandas()
blf_deltaspectral.set_index('id', inplace=True)

blf_spectral  = dt.fread(file_blf_spectral).to_pandas()
blf_spectral.set_index('id', inplace=True)

blf_spectralcontrast  = dt.fread(file_blf_spectralcontrast).to_pandas()
blf_spectralcontrast.set_index('id', inplace=True)

blf_vardeltaspectral  = dt.fread(file_blf_vardeltaspectral).to_pandas()
blf_vardeltaspectral.set_index('id', inplace=True)

essentia  = dt.fread(file_essentia).to_pandas()
essentia.set_index('id', inplace=True)

incp  = dt.fread(file_incp).to_pandas()
incp.set_index('id', inplace=True)

mfcc_bow  = dt.fread(file_mfcc_bow).to_pandas()
mfcc_bow.set_index('id', inplace=True)

mfcc_stats  = dt.fread(file_mfcc_stats).to_pandas()
mfcc_stats.set_index('id', inplace=True)

resnet  = dt.fread(file_resnet).to_pandas()
resnet.set_index('id', inplace=True)

vgg19  = dt.fread(file_vgg19).to_pandas()
vgg19.set_index('id', inplace=True)

blf_logfluc  = dt.fread(file_blf_logfluc)
new_cols = ['id']
new_cols.extend(list(blf_logfluc.names[2:]))
new_cols = tuple(new_cols)
del blf_logfluc[:, -1]
blf_logfluc.names = new_cols
blf_logfluc = blf_logfluc.to_pandas()
blf_logfluc.set_index('id', inplace=True)

print("Data Loaded....")
dataVectorsMapping = {
    "tfidf" : tf_idf,
    "bert"  : bert,
    "word2vec" : word2vec,
    "mfcc_bow" : mfcc_bow,
    "mfcc_stats" : mfcc_stats,
    "essentia" : essentia,
    "incp" : incp,
    "resnet" : resnet,
    "vgg19" : vgg19,
    "blf_delta_spectral" : blf_deltaspectral,
    "blf_correlation" : blf_correlation,
    "blf_logfluc" : blf_logfluc,
    "blf_spectral" : blf_spectral,
    "blf_spectral_contrast" : blf_spectralcontrast,
    "blf_vardelta_spectral" : blf_vardeltaspectral,  
}

class DataVector(Enum):
    tfidf = "tf_idf"
    bert = "bert"
    word2vec = "word2vec"
    mfcc_bow = "mfcc_bow"
    mfcc_stats = "mfcc_stats"
    essentia = "essentia"
    incp = "incp"
    resnet = "resnet"
    vgg19 = "vgg19"
    blf_delta_spectral = "blf_delta_spectral"
    blf_correlation = "blf_correlation"
    blf_logfluc = "blf_logfluc"
    blf_spectral = "blf_spectral"
    blf_spectral_contrast = "blf_spectral_contrast"
    blf_vardelta_spectral = "blf_vardelta_spectral"  

    def to_df(self):
        return dataVectorsMapping[self.name]


similarityFunctionMapping = {
    "cosineSim" : get_cosine_similarity,
    # "innerProduct" : get_innerProduct_similarity,
    "jaccardSim" : get_jaccard_similarity
}

class SimilarityFunction(Enum):
    cosineSim = "cosineSim"  # "Cosine Similarity"
    # innerProduct = "innerProduct"  # "Inner Product Similarity"
    jaccardSim = "jaccardSim"  # "Jaccard Similarity"

    def to_func(self):
        return similarityFunctionMapping[self.name]

def loadData(file):
    if exists(file):
        return pd.read_csv(file ,index_col=0)
    else:
        return pd.DataFrame( columns=range(100))

# Cosine
top_cosine_tfidf = loadData(f_top_cosine_tfidf)
top_cosine_word2vec = loadData(f_top_cosine_word2vec)
top_cosine_bert = loadData(f_top_cosine_bert)
top_cosine_mfcc_bow = loadData(f_top_cosine_mfcc_bow)
top_cosine_mfcc_stats = loadData(f_top_cosine_mfcc_stats)
top_cosine_essentia = loadData(f_top_cosine_essentia)
top_cosine_blf_delta_spectral = loadData(f_top_cosine_blf_delta_spectral)
top_cosine_blf_correlation = loadData(f_top_cosine_blf_correlation)
top_cosine_blf_logfluc = loadData(f_top_cosine_blf_logfluc)
top_cosine_blf_spectral = loadData(f_top_cosine_blf_spectral)
top_cosine_blf_spectral_contrast = loadData(f_top_cosine_blf_spectral_contrast)
top_cosine_blf_vardelta_spectral = loadData(f_top_cosine_blf_vardelta_spectral)
top_cosine_incp = loadData(f_top_cosine_incp)
top_cosine_vgg19 = loadData(f_top_cosine_vgg19)
top_cosine_resnet = loadData(f_top_cosine_resnet)

# Jaccard
top_jaccard_tfidf = loadData(f_top_jaccard_tfidf)
top_jaccard_word2vec = loadData(f_top_jaccard_word2vec)
top_jaccard_bert = loadData(f_top_jaccard_bert)
top_jaccard_mfcc_bow = loadData(f_top_jaccard_mfcc_bow)
top_jaccard_mfcc_stats = loadData(f_top_jaccard_mfcc_stats)
top_jaccard_essentia = loadData(f_top_jaccard_essentia)
top_jaccard_blf_delta_spectral = loadData(f_top_jaccard_blf_delta_spectral)
top_jaccard_blf_correlation = loadData(f_top_jaccard_blf_correlation)
top_jaccard_blf_logfluc = loadData(f_top_jaccard_blf_logfluc)
top_jaccard_blf_spectral = loadData(f_top_jaccard_blf_spectral)
top_jaccard_blf_spectral_contrast = loadData(f_top_jaccard_blf_spectral_contrast)
top_jaccard_blf_vardelta_spectral = loadData(f_top_jaccard_blf_vardelta_spectral)
top_jaccard_incp = loadData(f_top_jaccard_incp)
top_jaccard_vgg19 = loadData(f_top_jaccard_vgg19)
top_jaccard_resnet = loadData(f_top_jaccard_resnet)



# Inner Product

topIdsFiles = {
    "cosineSim_tfidf" : top_cosine_tfidf,
    "cosineSim_word2vec" : top_cosine_word2vec,
    "cosineSim_bert" : top_cosine_bert,
    "cosineSim_mfcc_bow": top_cosine_mfcc_bow,
    "cosineSim_mfcc_stats" : top_cosine_mfcc_stats,
    "cosineSim_essentia" : top_cosine_essentia,
    "cosineSim_blf_delta_spectral" : top_cosine_blf_delta_spectral,
    "cosineSim_blf_correlation" : top_cosine_blf_correlation,
    "cosineSim_blf_logfluc" : top_cosine_blf_logfluc,
    "cosineSim_blf_spectral" : top_cosine_blf_spectral,
    "cosineSim_blf_spectral_contrast" : top_cosine_blf_spectral_contrast,
    "cosineSim_blf_vardelta_spectral" : top_cosine_blf_vardelta_spectral,
    "cosineSim_incp" : top_cosine_incp,
    "cosineSim_vgg19" : top_cosine_vgg19,
    "cosineSim_resnet" : top_cosine_resnet,

    "jaccardSim_tfidf" : top_jaccard_tfidf,
    "jaccardSim_word2vec" : top_jaccard_word2vec,
    "jaccardSim_bert" : top_jaccard_bert,
    "jaccardSim_mfcc_bow": top_jaccard_mfcc_bow,
    "jaccardSim_mfcc_stats" : top_jaccard_mfcc_stats,
    "jaccardSim_essentia" : top_jaccard_essentia,
    "jaccardSim_blf_delta_spectral" : top_jaccard_blf_delta_spectral,
    "jaccardSim_blf_correlation" : top_jaccard_blf_correlation,
    "jaccardSim_blf_logfluc" : top_jaccard_blf_logfluc,
    "jaccardSim_blf_spectral" : top_jaccard_blf_spectral,
    "jaccardSim_blf_spectral_contrast" : top_jaccard_blf_spectral_contrast,
    "jaccardSim_blf_vardelta_spectral" : top_jaccard_blf_vardelta_spectral,
    "jaccardSim_incp" : top_jaccard_incp,
    "jaccardSim_vgg19" : top_jaccard_vgg19,
    "jaccardSim_resnet" : top_jaccard_resnet,

    # "innerProduct_tfidf" : top_innerProduct_tfidf,
    # "innerProduct_word2vec" : top_innerProduct_word2vec,
    # "innerProduct_bert" : top_innerProduct_bert,
}

csv_topIdsFiles = {
    "cosineSim_tfidf" : f_top_cosine_tfidf,
    "cosineSim_word2vec" : f_top_cosine_word2vec,
    "cosineSim_bert" : f_top_cosine_bert,
    "cosineSim_mfcc_bow": f_top_cosine_mfcc_bow,
    "cosineSim_mfcc_stats" : f_top_cosine_mfcc_stats,
    "cosineSim_essentia" : f_top_cosine_essentia,
    "cosineSim_blf_delta_spectral" : f_top_cosine_blf_delta_spectral,
    "cosineSim_blf_correlation" : f_top_cosine_blf_correlation,
    "cosineSim_blf_logfluc" : f_top_cosine_blf_logfluc,
    "cosineSim_blf_spectral" : f_top_cosine_blf_spectral,
    "cosineSim_blf_spectral_contrast" : f_top_cosine_blf_spectral_contrast,
    "cosineSim_blf_vardelta_spectral" : f_top_cosine_blf_vardelta_spectral,
    "cosineSim_incp" : f_top_cosine_incp,
    "cosineSim_vgg19" : f_top_cosine_vgg19,
    "cosineSim_resnet" : f_top_cosine_resnet,

    "jaccardSim_tfidf" : f_top_jaccard_tfidf,
    "jaccardSim_word2vec" : f_top_jaccard_word2vec,
    "jaccardSim_bert" : f_top_jaccard_bert,
    "jaccardSim_mfcc_bow": f_top_jaccard_mfcc_bow,
    "jaccardSim_mfcc_stats" : f_top_jaccard_mfcc_stats,
    "jaccardSim_essentia" : f_top_jaccard_essentia,
    "jaccardSim_blf_delta_spectral" : f_top_jaccard_blf_delta_spectral,
    "jaccardSim_blf_correlation" : f_top_jaccard_blf_correlation,
    "jaccardSim_blf_logfluc" : f_top_jaccard_blf_logfluc,
    "jaccardSim_blf_spectral" : f_top_jaccard_blf_spectral,
    "jaccardSim_blf_spectral_contrast" : f_top_jaccard_blf_spectral_contrast,
    "jaccardSim_blf_vardelta_spectral" : f_top_jaccard_blf_vardelta_spectral,
    "jaccardSim_incp" : f_top_jaccard_incp,
    "jaccardSim_vgg19" : f_top_jaccard_vgg19,
    "jaccardSim_resnet" : f_top_jaccard_resnet,

    # "innerProduct_tfidf" : './topIds/top_ids_innerProduct_tfidf_complete.csv',
    # "innerProduct_word2vec" : './topIds/top_ids_innerProduct_word2vec_complete.csv',
    # "innerProduct_bert" : './topIds/top_ids_innerProduct_bert_complete.csv',
    
}


app = FastAPI()

@app.get("/query/")
async def getTopResults(artist: str, track: str, top: int, vectorData: DataVector, simFunction: SimilarityFunction):
    print("\n\nGet Top results for \n\tArtist: ", artist, " \n\tTrack:",track, "\nUsing \n\tvectorData: ", vectorData.name, "\n\tsimilarity function: ", simFunction.name, "\n\tTop: ", top)

    id_song = getSongIdByQuery(artist, track)
    if id_song == None:
        return { "error" : "No record found for this query"}

    print("\nId song:", id_song)

    file_id = simFunction.name + "_" + vectorData.name

    if id_song in  topIdsFiles[file_id].index.values:
        # topValues = getTopIds(id_song,  topIdsFiles[file_id])
        print('\nQuery already in Top ids file for', file_id)
    else:
        print('\nNew song, calculating top 100 similar songs and saving to data')
        distances = compute_in_batches_distance(
            vectorData.to_df().to_numpy(),
            vectorData.to_df().loc[[id_song]].to_numpy(), 
            simfunction = simFunction.to_func(), 
            batches=1
            )

        indexValues =  vectorData.to_df().index
        location_self_distance = indexValues.get_indexer([id_song])[0]
        distances[location_self_distance] = -1
 
        print("Actual number of records:", len(topIdsFiles[file_id].index))
        # Add new record to top id file

        topIdsFiles[file_id].loc[id_song] = compute_topIds(distances.T, indexValues.values ,top=100)

        # print("New",topIdsFiles[file_id].loc[id_song])
        print("New number of records:", len(topIdsFiles[file_id].index))
        print("Writting to:", csv_topIdsFiles[file_id])

        # Update top id file in csv
        topIdsFiles[file_id].to_csv(csv_topIdsFiles[file_id])

    query_song = genres.loc[[id_song]].join(info, on="id")

    top_n_ids = topIdsFiles[file_id].loc[id_song].values[:top]

    topVal = genres.loc[top_n_ids].join(info, on="id")

    pk, mrrk, ndcgk = getMetrics(topIdsFiles[file_id].loc[[id_song]], top, genres)
    print("MAP@"+str(top), pk, "MRR@"+str(top), mrrk, "Mean NDCG@"+str(top), ndcgk, "\n\n")

    return { "song": query_song , "top": topVal, "metrics" : { "MAP" : pk, "MRR" : mrrk, "NDCG" : ndcgk }  }


@app.get("/metrics/")
async def getEvaluationMetrics(vectorData: DataVector, simFunction: SimilarityFunction, k:int):
    print("\nGet metrics for vectorData:", vectorData, "\nUsing similarity funtion", simFunction.name, "\nk:", k)

    file_id = simFunction.name + "_" + vectorData.name
    pk, mrrk, ndcgk = getMetrics(topIdsFiles[file_id], k, genres)

    print("\nMAP@"+str(k), pk, "\nMRR@"+str(k), mrrk, "\nMean NDCG@"+str(k), ndcgk)

    return { "SIM": simFunction.to_func(), "vectorData": vectorData.to_df(), "MAP" : pk, "MRR" : mrrk, "NDCG" : ndcgk }