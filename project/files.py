USE_COMPLETE_DATASETS = True

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
    f_top_baseline          = './data/top_ids_baseline_complete.csv'
    f_top_cosine_tfidf      = './data/top_ids_cosine_tfidf.csv'
    f_top_cosine_word2vec   = './data/top_ids_cosine_word2vec.csv'
    f_top_cosine_bert       = './data/top_ids_cosine_bert.csv'
    f_top_cosine_mfcc_bow   = './data/top_ids_cosine_mfcc_bow.csv'
    f_top_cosine_mfcc_stats = './data/top_ids_cosine_mfcc_stats.csv'
    f_top_cosine_essentia   = './data/top_ids_cosine_essentia.csv'
    f_top_cosine_incp       = './data/top_ids_cosine_incp.csv'
    f_top_cosine_resnet     = './data/top_ids_cosine_resnet.csv'
    f_top_cosine_vgg19      = './data/top_ids_cosine_vgg19.csv'
    f_top_cosine_blf_delta_spectral     = './data/top_ids_cosine_blf_delta_spectral.csv'
    f_top_cosine_blf_correlation        = './data/top_ids_cosine_blf_correlation.csv'
    f_top_cosine_blf_logfluc            = './data/top_ids_cosine_blf_logfluc.csv'
    f_top_cosine_blf_spectral           = './data/top_ids_cosine_blf_spectral.csv'
    f_top_cosine_blf_spectral_contrast  = './data/top_ids_cosine_blf_spectral_contrast.csv'
    f_top_cosine_blf_vardelta_spectral  = './data/top_ids_cosine_blf_vardelta_spectral.csv'

    f_top_jaccard_tfidf      = './data/top_ids_jaccard_tfidf.csv'
    f_top_jaccard_word2vec   = './data/top_ids_jaccard_word2vec.csv'
    f_top_jaccard_bert       = './data/top_ids_jaccard_bert.csv'
    f_top_jaccard_mfcc_bow   = './data/top_ids_jaccard_mfcc_bow.csv'
    f_top_jaccard_mfcc_stats = './data/top_ids_jaccard_mfcc_stats.csv'
    f_top_jaccard_essentia   = './data/top_ids_jaccard_essentia.csv'
    f_top_jaccard_incp       = './data/top_ids_jaccard_incp.csv'
    f_top_jaccard_resnet     = './data/top_ids_jaccard_resnet.csv'
    f_top_jaccard_vgg19      = './data/top_ids_jaccard_vgg19.csv'
    f_top_jaccard_blf_delta_spectral = './data/top_ids_jaccard_blf_delta_spectral.csv'
    f_top_jaccard_blf_correlation    = './data/top_ids_jaccard_blf_correlation.csv'
    f_top_jaccard_blf_logfluc        = './data/top_ids_jaccard_blf_logfluc.csv'
    f_top_jaccard_blf_spectral           = './data/top_ids_jaccard_blf_spectral.csv'
    f_top_jaccard_blf_spectral_contrast  = './data/top_ids_jaccard_blf_spectral_contrast.csv'
    f_top_jaccard_blf_vardelta_spectral  = './data/top_ids_jaccard_blf_vardelta_spectral.csv'
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
    f_top_cosine_blf_deltaspectral = './TopIdsTaskGenerated/top_ids_cosine_blf_deltaspectral_complete.csv'
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


# EARLY FUSION
f_top_cosine_bert_blf_spectral_resnet = "./data/top_ids_cosine_earlyfusion_bert_blf_spectral_resnet.csv"
f_top_cosine_bert_blf_spectral_incp   = "./data/top_ids_cosine_earlyfusion_bert_blf_spectral_incp.csv"
f_top_cosine_bert_mfcc_bow_resnet     = "./data/top_ids_cosine_earlyfusion_bert_mfcc_bow_resnet.csv"
f_top_cosine_bert_mfcc_bow_incp       = "./data/top_ids_cosine_earlyfusion_bert_mfcc_bow_incp.csv"
f_top_cosine_bert_blf_logfluc_resnet  = "./data/top_ids_cosine_earlyfusion_bert_blf_logfluc_resnet.csv"
f_top_cosine_bert_blf_logfluc_incp    = "./data/top_ids_cosine_earlyfusion_bert_blf_logfluc_incp.csv"
f_top_cosine_tfidf_blf_spectral_resnet= "./data/top_ids_cosine_earlyfusion_tfidf_blf_spectral_resnet.csv"
f_top_cosine_tfidf_blf_spectral_incp  = "./data/top_ids_cosine_earlyfusion_tfidf_blf_spectral_incp.csv"
f_top_cosine_tfidf_mfcc_bow_resnet    = "./data/top_ids_cosine_earlyfusion_tfidf_mfcc_bow_resnet.csv"
f_top_cosine_tfidf_mfcc_bow_incp      = "./data/top_ids_cosine_earlyfusion_tfidf_mfcc_bow_incp.csv"
f_top_cosine_tfidf_blf_logfluc_resnet = "./data/top_ids_cosine_earlyfusion_tfidf_blf_logfluc_resnet.csv"
f_top_cosine_tfidf_blf_logfluc_incp   = "./data/top_ids_cosine_earlyfusion_tfidf_blf_logfluc_incp.csv"

f_top_jaccard_bert_essentia_vgg19      = "./data/top_ids_jaccard_earlyfusion_bert_essentia_vgg19.csv"
f_top_jaccard_bert_essentia_resnet     = "./data/top_ids_jaccard_earlyfusion_bert_essentia_resnet.csv"
f_top_jaccard_bert_blf_logfluc_vgg19   = "./data/top_ids_jaccard_earlyfusion_bert_blf_logfluc_vgg19.csv"
f_top_jaccard_bert_blf_logfluc_resnet  = "./data/top_ids_jaccard_earlyfusion_bert_blf_logfluc_resnet.csv"
f_top_jaccard_bert_mfcc_stats_vgg19    = "./data/top_ids_jaccard_earlyfusion_bert_mfcc_stats_vgg19.csv"
f_top_jaccard_bert_mfcc_stats_resnet   = "./data/top_ids_jaccard_earlyfusion_bert_mfcc_stats_resnet.csv"
f_top_jaccard_tfidf_essentia_vgg19     = "./data/top_ids_jaccard_earlyfusion_tfidf_essentia_vgg19.csv"
f_top_jaccard_tfidf_essentia_resnet    = "./data/top_ids_jaccard_earlyfusion_tfidf_essentia_resnet.csv"
f_top_jaccard_tfidf_blf_logfluc_vgg19  = "./data/top_ids_jaccard_earlyfusion_tfidf_blf_logfluc_vgg19.csv"
f_top_jaccard_tfidf_blf_logfluc_resnet = "./data/top_ids_jaccard_earlyfusion_tfidf_blf_logfluc_resnet.csv"
f_top_jaccard_tfidf_mfcc_stats_vgg19   = "./data/top_ids_jaccard_earlyfusion_tfidf_mfcc_stats_vgg19.csv"
f_top_jaccard_tfidf_mfcc_stats_resnet  = "./data/top_ids_jaccard_earlyfusion_tfidf_mfcc_stats_resnet.csv"

# LATE FUSION
f_top_late_cosine_bert_blf_spectral_resnet = "./TopIdsFusion/top_ids_cosine_latefusion_bert_blf_spectral_resnet_complete.csv"
f_top_late_cosine_bert_blf_spectral_incp   = "./TopIdsFusion/top_ids_cosine_latefusion_bert_blf_spectral_incp_complete.csv"
f_top_late_cosine_bert_mfcc_bow_resnet     = "./TopIdsFusion/top_ids_cosine_latefusion_bert_mfcc_bow_resnet_complete.csv"
f_top_late_cosine_bert_mfcc_bow_incp       = "./TopIdsFusion/top_ids_cosine_latefusion_bert_mfcc_bow_incp_complete.csv"
f_top_late_cosine_bert_blf_logfluc_resnet  = "./TopIdsFusion/top_ids_cosine_latefusion_bert_blf_logfluc_resnet_complete.csv"
f_top_late_cosine_bert_blf_logfluc_incp    = "./TopIdsFusion/top_ids_cosine_latefusion_bert_blf_logfluc_incp_complete.csv"
f_top_late_cosine_tfidf_blf_spectral_resnet= "./TopIdsFusion/top_ids_cosine_latefusion_tfidf_blf_spectral_resnet_complete.csv"
f_top_late_cosine_tfidf_blf_spectral_incp  = "./TopIdsFusion/top_ids_cosine_latefusion_tfidf_blf_spectral_incp_complete.csv"
f_top_late_cosine_tfidf_mfcc_bow_resnet    = "./TopIdsFusion/top_ids_cosine_latefusion_tfidf_mfcc_bow_resnet_complete.csv"
f_top_late_cosine_tfidf_mfcc_bow_incp      = "./TopIdsFusion/top_ids_cosine_latefusion_tfidf_mfcc_bow_incp_complete.csv"
f_top_late_cosine_tfidf_blf_logfluc_resnet = "./TopIdsFusion/top_ids_cosine_latefusion_tfidf_blf_logfluc_resnet_complete.csv"
f_top_late_cosine_tfidf_blf_logfluc_incp   = "./TopIdsFusion/top_ids_cosine_latefusion_tfidf_blf_logfluc_incp_complete.csv"

f_top_late_jaccard_bert_essentia_vgg19      = "./TopIdsFusion/top_ids_jaccard_latefusion_bert_essentia_vgg19_complete.csv"
f_top_late_jaccard_bert_essentia_resnet     = "./TopIdsFusion/top_ids_jaccard_latefusion_bert_essentia_resnet_complete.csv"
f_top_late_jaccard_bert_blf_logfluc_vgg19   = "./TopIdsFusion/top_ids_jaccard_latefusion_bert_blf_logfluc_vgg19_complete.csv"
f_top_late_jaccard_bert_blf_logfluc_resnet  = "./TopIdsFusion/top_ids_jaccard_latefusion_bert_blf_logfluc_resnet_complete.csv"
f_top_late_jaccard_bert_mfcc_stats_vgg19    = "./TopIdsFusion/top_ids_jaccard_latefusion_bert_mfcc_stats_vgg19_complete.csv"
f_top_late_jaccard_bert_mfcc_stats_resnet   = "./TopIdsFusion/top_ids_jaccard_latefusion_bert_mfcc_stats_resnet_complete.csv"
f_top_late_jaccard_tfidf_essentia_vgg19     = "./TopIdsFusion/top_ids_jaccard_latefusion_tfidf_essentia_vgg19_complete.csv"
f_top_late_jaccard_tfidf_essentia_resnet    = "./TopIdsFusion/top_ids_jaccard_latefusion_tfidf_essentia_resnet_complete.csv"
f_top_late_jaccard_tfidf_blf_logfluc_vgg19  = "./TopIdsFusion/top_ids_jaccard_latefusion_tfidf_blf_logfluc_vgg19_complete.csv"
f_top_late_jaccard_tfidf_blf_logfluc_resnet = "./TopIdsFusion/top_ids_jaccard_latefusion_tfidf_blf_logfluc_resnet_complete.csv"
f_top_late_jaccard_tfidf_mfcc_stats_vgg19   = "./TopIdsFusion/top_ids_jaccard_latefusion_tfidf_mfcc_stats_vgg19_complete.csv"
f_top_late_jaccard_tfidf_mfcc_stats_resnet  = "./TopIdsFusion/top_ids_jaccard_latefusion_tfidf_mfcc_stats_resnet_complete.csv"

# Files metrics

file_metrics_cosine = './TopIdsFusion/df_metrics_cosine_individual.csv'
file_metrics_jaccard = './TopIdsFusion/df_metrics_jaccard_individual.csv'

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

# Task 3
file_metrics_earlyfusion_jaccard = './TopIdsFusion//df_metric_earlyfusion_jaccard.csv'
file_metrics_earlyfusion_cosine =  './TopIdsFusion/df_metrics_earlyfusion_cosine.csv'

file_metrics_latefusion_jaccard = './TopIdsFusion//df_metric_latefusion_jaccard.csv'
file_metrics_latefusion_cosine =  './TopIdsFusion/df_metrics_latefusion_cosine.csv'

file_cosine_early_mean_precision_datasets         = './TopIdsFusion/df_cosine_early_mean_precision_datasets_plot.csv'
file_cosine_early_mean_recall_datasets            = './TopIdsFusion/df_cosine_early_mean_recall_datasets_plot.csv'
file_cosine_early_maxprecision_precision_datasets = './TopIdsFusion/df_cosine_early_mean_maxprecision_datasets_plot.csv'

file_jaccard_early_mean_precision_datasets         = './TopIdsFusion/df_jaccard_early_mean_precision_datasets_plot.csv'
file_jaccard_early_mean_recall_datasets            = './TopIdsFusion/df_jaccard_early_mean_recall_datasets_plot.csv'
file_jaccard_early_maxprecision_precision_datasets = './TopIdsFusion/df_jaccard_early_mean_maxprecision_datasets_plot.csv'

file_cosine_late_mean_precision_datasets         = './TopIdsFusion/df_cosine_late_mean_precision_datasets_plot.csv'
file_cosine_late_mean_recall_datasets            = './TopIdsFusion/df_cosine_late_mean_recall_datasets_plot.csv'
file_cosine_late_maxprecision_precision_datasets = './TopIdsFusion/df_cosine_late_mean_maxprecision_datasets_plot.csv'

file_jaccard_late_mean_precision_datasets         = './TopIdsFusion/df_jaccard_late_mean_precision_datasets_plot.csv'
file_jaccard_late_mean_recall_datasets            = './TopIdsFusion/df_jaccard_late_mean_recall_datasets_plot.csv'
file_jaccard_late_maxprecision_precision_datasets = './TopIdsFusion/df_jaccard_late_mean_maxprecision_datasets_plot.csv'