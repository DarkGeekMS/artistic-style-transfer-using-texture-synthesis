import numpy as np
from sklearn.feature_extraction.image import extract_patches
from ml_models.pca import transform_pca

def IRLS(X, X_patches, style_patches, neighbors, proj_matrix, patch_size, sub_gap, I_irls, r):
    """ 
    The function to perform IRLS for least square optimization
    between style and stylized image patches. 

    Parameters: 
    X (ndarray): the initial estimate of stylized image ----> Beta
    X_patches (ndarray): array of patches from the stylized image X ----> X
    style_patches (ndarray): array of style patches (of the same resolution as current X) ----> Y 
    neighbors (Object): sklearn KNN object  
    proj_matrix (ndarray): pca projection matrix for x patches reduction
    I_irls (int): number of IRLS iterations
    r (int): the robust statistic value

    Returns: 
    ndarray: the temporal estimate Xtilde

    """   
    for iter_1 in range(I_irls): # loop over irls iterations
        # get X features and project them
        x_features = X_patches.reshape(-1, X_patches.shape[3] * X_patches.shape[3] * 3)
        x_features = transform_pca(proj_matrix, x_features)
        style_patches_iter = style_patches.reshape(-1, style_patches.shape[3], style_patches.shape[4], style_patches.shape[5])
        
        # apply KNN and initialize weights matrix
        distances, indices = neighbors.kneighbors(x_features)
        distances += 0.0001
        W = np.power(distances, r-2)

        # initialize the cummulative matrix and extract patches corresponding to both X and style patches
        cum_mat = np.zeros((X.shape[0], X.shape[0], 3), dtype=np.float32)
        cum_patches = extract_patches(cum_mat, patch_shape=(patch_size, patch_size, 3), extraction_step=sub_gap)

        # reset X
        X[:] = 0

        cum_idx = 0
        for iter_2 in range(X_patches.shape[0]): # loop over all X patches
            for iter_3 in range(X_patches.shape[1]):
                nearest_neighbor = style_patches_iter[indices[cum_idx, 0]] # find the nearest neighbor for that single patch / pick only one
                X_patches[iter_2, iter_3, 0, :, :, :] += nearest_neighbor * W[cum_idx] # update patch based on neighbor and weights
                cum_patches[iter_2, iter_3, 0, :, :, :] += 1 * W[cum_idx] # update cummulative weights 
                cum_idx = cum_idx + 1

        cum_mat += 0.0001 # non-zero division
        X /= cum_mat

    return X