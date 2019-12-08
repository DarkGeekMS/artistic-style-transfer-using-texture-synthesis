import numpy as np
from numpy.linalg import eig
import skimage.io as io


def pca(original_feature_vector):
    """
    Dimensions reduction function that take a feature vector and project it after extracting the principle components
    Args:
        original_feature_vector: the feature vector of the flattened patches

    Returns:
        projection_matrix: the projection matrix used to transform the original feature vectors
        result: the result of the matrix multiplication = Row of the projection matrix *
         Row of the adjusted data(Centralized)
    """
    # calculate the mean of each Feature column
    mean_feature_vector = np.mean(original_feature_vector, axis=0)
    # center feature columns by subtracting column means(Data adjusted)
    centralized_matrix = original_feature_vector - mean_feature_vector
    # calculate covariance matrix of centered matrix (size of the covariance matrix = (features,features))
    covariance_matrix = np.cov(centralized_matrix.T)
    # eigen decomposition of covariance matrix
    values, projection_matrix = eig(covariance_matrix)
    projection_matrix = projection_matrix.real
    # projected data
    result = projection_matrix.T.dot(centralized_matrix.T)

    return projection_matrix, result.T


def transform_pca(projection_matrix, original_feature_vector):
    """
    this function takes the projection matrix and the feature vecter and multiply them
    after adjusting the data by subtracting the mean.
    Args:
        projection_matrix: the projection matrix used to project the original data
        original_feature_vector: the feature vector of the flattened patches

    Returns:
        result: after projecting the data
    """
    # calculate the mean of each Feature column
    mean_feature_vector = np.mean(original_feature_vector, axis=0)
    # center feature columns by subtracting column means(Data adjusted)
    centralized_matrix = original_feature_vector - mean_feature_vector
    # multiplication between rows of the projection matrix and the rows of the feature vector after adjusting it
    result = projection_matrix.T.dot(centralized_matrix.T)
    return result.T
