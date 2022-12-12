USE_COMPLETE_DATASETS = False

# DATA FILES PROVIDED Task1
file_tfidf = "./../MMSR_WT22_Task1_Data/id_lyrics_tf-idf_mmsr.tsv"
file_word2vec = "./../MMSR_WT22_Task1_Data/id_lyrics_word2vec_mmsr.tsv"
file_bert = "./../MMSR_WT22_Task1_Data/id_bert_mmsr.tsv"
file_genres = "./../MMSR_WT22_Task1_Data/id_genres_mmsr.tsv"
file_info = "./../MMSR_WT22_Task1_Data/id_information_mmsr.tsv"

# DATA FILES PROVIDED Task 2
file_genres_2   = "./../task2/id_genres_mmsr.tsv"
file_info_2     = "./../task2/id_information_mmsr.tsv"

# EMD Layer, Lyrics Features
file_tfidf_2    = "./../task2/id_lyrics_tf-idf_mmsr.tsv"
file_word2vec_2 = "./../task2/id_lyrics_word2vec_mmsr.tsv"
file_bert_2     = "./../task2/id_lyrics_bert_mmsr.tsv"

## Audio Layer, Acoustic Features
### Block Level features (BLFs)
file_blf_correlation      = "./../task2/id_blf_correlation_mmsr.tsv"
file_blf_deltaspectral    = "./../task2/id_blf_deltaspectral_mmsr.tsv"
file_blf_logfluc          = "./../task2/id_blf_logfluc_mmsr.tsv"
file_blf_spectral         = "./../task2/id_blf_spectral_mmsr.tsv"
file_blf_spectralcontrast = "./../task2/id_blf_spectralcontrast_mmsr.tsv"
file_blf_vardeltaspectral = "./../task2/id_blf_vardeltaspectral_mmsr.tsv"
#### Mel Frequency Cepstral Coefficients (MFFCs)
file_mfcc_bow   = "./../task2/id_mfcc_bow_mmsr.tsv"
file_mfcc_stats = "./../task2/id_mfcc_stats_mmsr.tsv"
file_essentia   = "./../task2/id_essentia_mmsr.tsv"

## UGC Layer , Video Features
## Derivative Content DC, Video Features
file_resnet   = "./../task2/id_resnet_mmsr.tsv"
file_vgg19    = "./../task2/id_vgg19_mmsr.tsv"
file_incp       = "./../task2/id_incp_mmsr.tsv"

# Generated Datasets

if USE_COMPLETE_DATASETS:
    # Individual Features with jaccard and cosine
    f_top_baseline          = './TopIdsTask2/top_ids_baseline_complete.csv'
    f_top_cosine_tfidf      = './TopIdsTask2/top_ids_cosine_tfidf_complete_.csv'
    f_top_cosine_word2vec   = './TopIdsTask2/top_ids_cosine_word2vec_complete_.csv'
    f_top_cosine_bert       = './TopIdsTask2/top_ids_cosine_bert_complete_.csv'
    f_top_cosine_mfcc_bow   = './TopIdsTask2/top_ids_cosine_mfcc_bow_complete_.csv'
    f_top_cosine_mfcc_stats = './TopIdsTask2/top_ids_cosine_mfcc_stats_complete_.csv'
    f_top_cosine_essentia   = './TopIdsTask2/top_ids_cosine_essentia_complete_.csv'
    f_top_cosine_incp       = './TopIdsTask2/top_ids_cosine_incp_complete_.csv'
    f_top_cosine_resnet     = './TopIdsTask2/top_ids_cosine_resnet_complete_.csv'
    f_top_cosine_vgg19      = './TopIdsTask2/top_ids_cosine_vgg19_complete_.csv'
    f_top_cosine_blf_delta_spectral = './TopIdsTask2/top_ids_cosine_blf_deltaspectral_complete_.csv'
    f_top_cosine_blf_correlation    = './TopIdsTask2/top_ids_cosine_blf_correlation_complete_.csv'
    f_top_cosine_blf_logfluc        = './TopIdsTask2/top_ids_cosine_blf_logfluc_complete_.csv'
    f_top_cosine_blf_spectral           = './TopIdsTask2/top_ids_cosine_blf_spectral_complete_.csv'
    f_top_cosine_blf_spectral_contrast  = './TopIdsTask2/top_ids_cosine_blf_spectralcontrast_complete_.csv'
    f_top_cosine_blf_vardelta_spectral  = './TopIdsTask2/top_ids_cosine_blf_vardeltaspectral_complete_.csv'

    f_top_jaccard_tfidf      = './TopIdsTask2/top_ids_jaccard_tfidf_complete.csv'
    f_top_jaccard_word2vec   = './TopIdsTask2/top_ids_jaccard_word2vec_complete.csv'
    f_top_jaccard_bert       = './TopIdsTask2/top_ids_jaccard_bert_complete.csv'
    f_top_jaccard_mfcc_bow   = './TopIdsTask2/top_ids_jaccard_mfcc_bow_complete.csv'
    f_top_jaccard_mfcc_stats = './TopIdsTask2/top_ids_jaccard_mfcc_stats_complete.csv'
    f_top_jaccard_essentia   = './TopIdsTask2/top_ids_jaccard_essentia_complete.csv'
    f_top_jaccard_incp       = './TopIdsTask2/top_ids_jaccard_incp_complete.csv'
    f_top_jaccard_resnet     = './TopIdsTask2/top_ids_jaccard_resnet_complete.csv'
    f_top_jaccard_vgg19      = './TopIdsTask2/top_ids_jaccard_vgg19_complete.csv'
    f_top_jaccard_blf_delta_spectral = './TopIdsTask2/top_ids_jaccard_blf_deltaspectral_complete.csv'
    f_top_jaccard_blf_correlation    = './TopIdsTask2/top_ids_jaccard_blf_correlation_complete.csv'
    f_top_jaccard_blf_logfluc        = './TopIdsTask2/top_ids_jaccard_blf_logfluc_complete.csv'
    f_top_jaccard_blf_spectral           = './TopIdsTask2/top_ids_jaccard_blf_spectral_complete.csv'
    f_top_jaccard_blf_spectral_contrast  = './TopIdsTask2/top_ids_jaccard_blf_spectralcontrast_complete.csv'
    f_top_jaccard_blf_vardelta_spectral  = './TopIdsTask2/top_ids_jaccard_blf_vardeltaspectral_complete.csv'
else:
    # Individual Features with jaccard and cosine NEWS TO TEST
    f_top_baseline          = './TopIdsTask2/top_ids_baseline_complete.csv'
    f_top_cosine_tfidf      = './TopIdsTaskGenerated/top_ids_cosine_tfidf_complete.csv'
    f_top_cosine_word2vec   = './TopIdsTaskGenerated/top_ids_cosine_word2vec_complete.csv'
    f_top_cosine_bert       = './TopIdsTaskGenerated/top_ids_cosine_bert_complete.csv'
    f_top_cosine_mfcc_bow   = './TopIdsTaskGenerated/top_ids_cosine_mfcc_bow_complete.csv'
    f_top_cosine_mfcc_stats = './TopIdsTaskGenerated/top_ids_cosine_mfcc_stats_complete.csv'
    f_top_cosine_essentia   = './TopIdsTaskGenerated/top_ids_cosine_essentia_complete.csv'
    f_top_cosine_incp       = './TopIdsTaskGenerated/top_ids_cosine_incp_complete.csv'
    f_top_cosine_resnet     = './TopIdsTaskGenerated/top_ids_cosine_resnet_complete.csv'
    f_top_cosine_vgg19      = './TopIdsTaskGenerated/top_ids_cosine_vgg19_complete.csv'
    f_top_cosine_blf_delta_spectral = './TopIdsTaskGenerated/top_ids_cosine_blf_deltaspectral_complete.csv'
    f_top_cosine_blf_correlation    = './TopIdsTaskGenerated/top_ids_cosine_blf_correlation_complete.csv'
    f_top_cosine_blf_logfluc        = './TopIdsTaskGenerated/top_ids_cosine_blf_logfluc_complete.csv'
    f_top_cosine_blf_spectral           = './TopIdsTaskGenerated/top_ids_cosine_blf_spectral_complete.csv'
    f_top_cosine_blf_spectral_contrast  = './TopIdsTaskGenerated/top_ids_cosine_blf_spectralcontrast_complete.csv'
    f_top_cosine_blf_vardelta_spectral  = './TopIdsTaskGenerated/top_ids_cosine_blf_vardeltaspectral_complete.csv'

    f_top_jaccard_tfidf      = './TopIdsTaskGenerated/top_ids_jaccard_tfidf_complete.csv'
    f_top_jaccard_word2vec   = './TopIdsTaskGenerated/top_ids_jaccard_word2vec_complete.csv'
    f_top_jaccard_bert       = './TopIdsTaskGenerated/top_ids_jaccard_bert_complete.csv'
    f_top_jaccard_mfcc_bow   = './TopIdsTaskGenerated/top_ids_jaccard_mfcc_bow_complete.csv'
    f_top_jaccard_mfcc_stats = './TopIdsTaskGenerated/top_ids_jaccard_mfcc_stats_complete.csv'
    f_top_jaccard_essentia   = './TopIdsTaskGenerated/top_ids_jaccard_essentia_complete.csv'
    f_top_jaccard_incp       = './TopIdsTaskGenerated/top_ids_jaccard_incp_complete.csv'
    f_top_jaccard_resnet     = './TopIdsTaskGenerated/top_ids_jaccard_resnet_complete.csv'
    f_top_jaccard_vgg19      = './TopIdsTaskGenerated/top_ids_jaccard_vgg19_complete.csv'
    f_top_jaccard_blf_delta_spectral = './TopIdsTaskGenerated/top_ids_jaccard_blf_deltaspectral_complete.csv'
    f_top_jaccard_blf_correlation    = './TopIdsTaskGenerated/top_ids_jaccard_blf_correlation_complete.csv'
    f_top_jaccard_blf_logfluc        = './TopIdsTaskGenerated/top_ids_jaccard_blf_logfluc_complete.csv'
    f_top_jaccard_blf_spectral           = './TopIdsTaskGenerated/top_ids_jaccard_blf_spectral_complete.csv'
    f_top_jaccard_blf_spectral_contrast  = './TopIdsTaskGenerated/top_ids_jaccard_blf_spectralcontrast_complete.csv'
    f_top_jaccard_blf_vardelta_spectral  = './TopIdsTaskGenerated/top_ids_jaccard_blf_vardeltaspectral_complete.csv'

# Features combined [Lyrics] [Audio] [Video] with Jaccard and cosine
f_top_cosine_tfidf_mfcc_bow_incp  = './TopIdsTask2/top_ids_cosine_tfidf_mfcc_bow_incp_complete.csv'
f_top_cosine_tfidf_mfcc_bow_vgg19 = './TopIdsTask2/top_ids_cosine_tfidf_mfcc_bow_vgg19_complete.csv'
f_top_cosine_tfidf_essentia_incp  = './TopIdsTask2/top_ids_cosine_tfidf_essentia_incp_complete.csv'
f_top_cosine_tfidf_essentia_vgg19 = './TopIdsTask2/top_ids_cosine_tfidf_essentia_vgg19_complete.csv'
f_top_cosine_tfidf_blf_delta_spectral_incp  = './TopIdsTask2/top_ids_cosine_tfidf_blfdeltaspectral_incp_complete.csv'
f_top_cosine_tfidf_blf_delta_spectral_vgg19 = './TopIdsTask2/top_ids_cosine_tfidf_blfdeltaspectral_vgg19_complete.csv'
f_top_cosine_bert_mfcc_bow_incp  = './TopIdsTask2/top_ids_cosine_bert_mfcc_bow_incp_complete.csv'
f_top_cosine_bert_mfcc_bow_vgg19 = './TopIdsTask2/top_ids_cosine_bert_mfcc_bow_vgg19_complete.csv'
f_top_cosine_bert_essentia_incp  = './TopIdsTask2/top_ids_cosine_bert_essentia_incp_complete.csv'
f_top_cosine_bert_essentia_vgg19 = './TopIdsTask2/top_ids_cosine_bert_essentia_vgg19_complete.csv'
f_top_cosine_bert_blf_delta_spectral_incp  = './TopIdsTask2/top_ids_cosine_bert_blfdeltaspectral_incp_complete.csv'
f_top_cosine_bert_blf_delta_spectral_vgg19 = './TopIdsTask2/top_ids_cosine_bert_blfdeltaspectral_vgg19_complete.csv'

f_top_jaccard_tfidf_mfcc_bow_incp  = './TopIdsTask2/top_ids_jaccard_tfidf_mfcc_bow_incp_complete.csv'
f_top_jaccard_tfidf_mfcc_bow_vgg19 = './TopIdsTask2/top_ids_jaccard_tfidf_mfcc_bow_vgg19_complete.csv'
f_top_jaccard_tfidf_essentia_incp  = './TopIdsTask2/top_ids_cosine_tfidf_essentia_incp_complete.csv'
f_top_jaccard_tfidf_essentia_vgg19 = './TopIdsTask2/top_ids_cosine_tfidf_essentia_vgg19_complete.csv'
f_top_jaccard_tfidf_blf_delta_spectral_incp = './TopIdsTask2/top_ids_jaccard_tfidf_blfdeltaspectral_incp_complete.csv'
f_top_jaccard_tfidf_blf_delta_spectral_vgg19 = './TopIdsTask2/top_ids_jaccard_tfidf_blfdeltaspectral_vgg19_complete.csv'
f_top_jaccard_bert_mfcc_bow_incp  = './TopIdsTask2/top_ids_jaccard_bert_mfcc_bow_incp_complete.csv'
f_top_jaccard_bert_mfcc_bow_vgg19 = './TopIdsTask2/top_ids_jaccard_bert_mfcc_bow_vgg19_complete.csv'
f_top_jaccard_bert_essentia_incp  = './TopIdsTask2/top_ids_jaccard_bert_essentia_incp_complete.csv'
f_top_jaccard_bert_essentia_vgg19 = './TopIdsTask2/top_ids_jaccard_bert_essentia_vgg19_complete.csv'
f_top_jaccard_bert_blf_delta_spectral_incp  = './TopIdsTask2/top_ids_jaccard_bert_blfdeltaspectral_incp_complete.csv'
f_top_jaccard_bert_blf_delta_spectral_vgg19 = './TopIdsTask2/top_ids_jaccard_bert_blfdeltaspectral_vgg19_complete.csv'

# Files metrics

file_metrics_jaccard = './TopIdsTask2/df_metrics_jaccard.csv'
file_metrics_cosine = './TopIdsTask2/df_metrics_cosine.csv'

file_cosine_mean_precision_datasets = './TopIdsTask2/df_cosine_mean_precision_datasets_plot.csv'
file_cosine_mean_recall_datasets = './TopIdsTask2/df_cosine_mean_recall_datasets_plot.csv'
file_cosine_maxprecision_precision_datasets = './TopIdsTask2/df_cosine_mean_maxprecision_datasets_plot.csv'

file_jaccard_mean_precision_datasets = './TopIdsTask2/df_jaccard_mean_precision_datasets_plot.csv'
file_jaccard_mean_recall_datasets = './TopIdsTask2/df_jaccard_mean_recall_datasets_plot.csv'
file_jaccard_maxprecision_precision_datasets = './TopIdsTask2/df_jaccard_mean_maxprecision_datasets_plot.csv'

file_corr_all_values_tau_cosine = './TopIdsTask2/corr_all_values_tau_cosine.csv'
file_correlations_cosine_tau = './TopIdsTask2/correlations_cosine_tau.csv'

file_corr_all_values_tau_jaccard = './TopIdsTask2/corr_all_values_tau_jaccard.csv'
file_correlations_jaccard_tau = './TopIdsTask2/correlations_jaccard_tau.csv'