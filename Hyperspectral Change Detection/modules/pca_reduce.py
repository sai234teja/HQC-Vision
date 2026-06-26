from sklearn.decomposition import PCA
import numpy as np

def apply_pca(t1, t2, n_components=3):
    h, w, c = t1.shape
    pca = PCA(n_components=n_components)
    
    # Fit on T1, transform both to maintain same feature space
    t1_reduced = pca.fit_transform(t1.reshape(-1, c)).reshape(h, w, n_components)
    t2_reduced = pca.transform(t2.reshape(-1, c)).reshape(h, w, n_components)
    
    return t1_reduced, t2_reduced