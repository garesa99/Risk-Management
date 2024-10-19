import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(returns):
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    pca = PCA()
    pca_results = pca.fit_transform(scaled_returns)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    return pca, pca_results, explained_variance_ratio, cumulative_variance_ratio

def get_factor_loadings(pca, returns, num_components):
    loadings = pd.DataFrame(
        pca.components_.T[:, :num_components],
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=returns.columns
    )
    return loadings

def biplot_data(pca, returns):
    scaled_returns = StandardScaler().fit_transform(returns)
    scores = pca.transform(scaled_returns)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return scores, loadings