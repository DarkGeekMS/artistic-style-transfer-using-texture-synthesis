import collections
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
    eigen_values, eigen_vectors = eig(covariance_matrix)
    # order eigen vectors descendingly according to eigen values
    eigen_pairs = {}
    for it in range(eigen_vectors.shape[0]):
        eigen_pairs[eigen_values[it]] = eigen_vectors[:, it]
    eigen_pairs = dict(collections.OrderedDict(sorted(eigen_pairs.items(), reverse=True)))
    i = 0
    for key in eigen_pairs:
        eigen_values[i] = key
        eigen_vectors[:, i] = eigen_pairs[key]
        i = i + 1
    # normalize eigen values
    eigen_values = eigen_values / np.sum(eigen_values)
    # accumulate eigen values to a certain threshold
    num = 0
    summation = 0
    for i in range(eigen_values.shape[0]):
        if summation >= 0.95:
            break
        num = num + 1
        summation = summation + eigen_values[i]
    # building the projection matrix from some normalized eigen vectors
    projection_matrix = np.zeros((num, centralized_matrix.shape[1]))
    for i in range(num):
        projection_matrix[i] = eigen_vectors[:, i].real
    # project data
    result = (np.matmul(projection_matrix, centralized_matrix.T)).T

    return projection_matrix, result


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
    result = (np.matmul(projection_matrix, centralized_matrix.T)).T
    return result
