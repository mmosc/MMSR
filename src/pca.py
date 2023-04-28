import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE

# PCA feature selection, we can get only features, so we still have 99% explained variance from the Data.
def get_PCA_selection(features: np.array, n_components=0.99) -> pd.DataFrame:
    pca = PCA(n_components=n_components, svd_solver="auto")
    return pd.DataFrame(pca.fit_transform(features), index=features.index)


# We can use also PCA Kernel, but we need to fix the number of components. Here we have different things to try
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
def get_PCA_Kernel_selection(
    features: np.array, kernel="rbf", n_components: int = 100, gamma=None
) -> pd.DataFrame:
    kernel_pca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    return pd.DataFrame(kernel_pca.fit_transform(features))


# Fast ICA would be another solution of dimensionality reduction
def get_ICA_selection(features: np.array, n_components: int = 100) -> pd.DataFrame:
    fast_ICA = FastICA(n_components=n_components)
    return pd.DataFrame(fast_ICA.fit_transform(features))


# We could also try TSNE, but not sure about performance. maybe long computation time
def get_TSNE_selection(features: np.array, n_components: int = 100) -> pd.DataFrame:
    tsne = TSNE(n_components=n_components, learning_rate="auto", init="random")
    return pd.DataFrame(tsne.fit_transform(features))
