"""
Distance metrics for comparing image descriptors
Implements various distance measures for visual search
"""

import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine, correlation


def cvpr_compare(F1, F2, distance_metric='euclidean'):
    """
    Compare two feature descriptors using specified distance metric.
    
    Args:
        F1: First feature descriptor (1D numpy array)
        F2: Second feature descriptor (1D numpy array)
        distance_metric: Type of distance metric to use
    
    Returns:
        Distance value (float)
    """
    # Ensure inputs are 1D arrays
    F1 = np.array(F1).flatten()
    F2 = np.array(F2).flatten()
    
    if distance_metric == 'euclidean':
        return euclidean_distance(F1, F2)
    elif distance_metric == 'l1' or distance_metric == 'manhattan':
        return l1_distance(F1, F2)
    elif distance_metric == 'cosine':
        return cosine_distance(F1, F2)
    elif distance_metric == 'correlation':
        return correlation_distance(F1, F2)
    elif distance_metric == 'chi_square':
        return chi_square_distance(F1, F2)
    elif distance_metric == 'intersection':
        return histogram_intersection_distance(F1, F2)
    elif distance_metric == 'mahalanobis':
        # Mahalanobis requires covariance matrix, handled separately
        raise ValueError("Mahalanobis distance requires covariance matrix. Use mahalanobis_distance() directly.")
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")


def euclidean_distance(F1, F2):
    """
    Compute Euclidean (L2) distance between two descriptors.
    
    Args:
        F1, F2: Feature descriptors
    
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((F1 - F2) ** 2))


def l1_distance(F1, F2):
    """
    Compute L1 (Manhattan) distance between two descriptors.
    
    Args:
        F1, F2: Feature descriptors
    
    Returns:
        L1 distance
    """
    return np.sum(np.abs(F1 - F2))


def cosine_distance(F1, F2):
    """
    Compute cosine distance between two descriptors.
    
    Args:
        F1, F2: Feature descriptors
    
    Returns:
        Cosine distance (1 - cosine similarity)
    """
    dot_product = np.dot(F1, F2)
    norm_F1 = np.linalg.norm(F1)
    norm_F2 = np.linalg.norm(F2)
    
    if norm_F1 == 0 or norm_F2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm_F1 * norm_F2)
    return 1.0 - cosine_sim


def correlation_distance(F1, F2):
    """
    Compute correlation distance between two descriptors.
    
    Args:
        F1, F2: Feature descriptors
    
    Returns:
        Correlation distance
    """
    F1_centered = F1 - np.mean(F1)
    F2_centered = F2 - np.mean(F2)
    
    correlation = np.dot(F1_centered, F2_centered) / (np.linalg.norm(F1_centered) * np.linalg.norm(F2_centered) + 1e-10)
    return 1.0 - correlation


def chi_square_distance(F1, F2):
    """
    Compute Chi-square distance between two histograms.
    Useful for histogram comparison.
    
    Args:
        F1, F2: Feature descriptors (histograms)
    
    Returns:
        Chi-square distance
    """
    # Avoid division by zero
    denominator = F1 + F2 + 1e-10
    numerator = (F1 - F2) ** 2
    
    return 0.5 * np.sum(numerator / denominator)


def histogram_intersection_distance(F1, F2):
    """
    Compute histogram intersection distance.
    Returns 1 - intersection (so smaller is more similar).
    
    Args:
        F1, F2: Feature descriptors (histograms)
    
    Returns:
        Histogram intersection distance
    """
    intersection = np.sum(np.minimum(F1, F2))
    return 1.0 - intersection


def mahalanobis_distance(F1, F2, cov_inv):
    """
    Compute Mahalanobis distance between two descriptors.
    
    Args:
        F1, F2: Feature descriptors
        cov_inv: Inverse of covariance matrix
    
    Returns:
        Mahalanobis distance
    """
    diff = F1 - F2
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))


def batch_compare(query_descriptor, all_descriptors, distance_metric='euclidean', cov_inv=None):
    """
    Efficiently compare query descriptor against all descriptors.
    
    Args:
        query_descriptor: Query feature descriptor
        all_descriptors: Matrix of all descriptors (N x D)
        distance_metric: Distance metric to use
        cov_inv: Inverse covariance matrix (for Mahalanobis)
    
    Returns:
        Array of distances
    """
    N = all_descriptors.shape[0]
    distances = np.zeros(N)
    
    if distance_metric == 'mahalanobis' and cov_inv is not None:
        # Vectorized Mahalanobis distance
        diff = all_descriptors - query_descriptor
        distances = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
    elif distance_metric == 'euclidean':
        # Vectorized Euclidean distance
        distances = np.sqrt(np.sum((all_descriptors - query_descriptor) ** 2, axis=1))
    elif distance_metric == 'l1':
        # Vectorized L1 distance
        distances = np.sum(np.abs(all_descriptors - query_descriptor), axis=1)
    elif distance_metric == 'chi_square':
        # Vectorized Chi-square distance
        denominator = all_descriptors + query_descriptor + 1e-10
        numerator = (all_descriptors - query_descriptor) ** 2
        distances = 0.5 * np.sum(numerator / denominator, axis=1)
    else:
        # Fall back to loop for other metrics
        for i in range(N):
            distances[i] = cvpr_compare(query_descriptor, all_descriptors[i], distance_metric)
    
    return distances

