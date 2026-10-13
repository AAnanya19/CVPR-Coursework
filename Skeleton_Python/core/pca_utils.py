"""
PCA dimensionality reduction module for visual search
Provides PCA transformation and projection utilities
"""

import numpy as np
import pickle
import os


class PCAProjection:
    """
    PCA projection class for dimensionality reduction.
    """
    
    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Initialize PCA projection.
        
        Args:
            n_components: Number of components to keep (None = auto from variance)
            variance_threshold: Cumulative variance to preserve (if n_components is None)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.cov_matrix = None
        self.cov_inv = None
        
    def fit(self, X):
        """
        Fit PCA to data.
        
        Args:
            X: Data matrix (N x D) where N is number of samples, D is dimensionality
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        self.cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Determine number of components
        if self.n_components is None:
            # Keep components that explain variance_threshold of variance
            cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
        
        # Store components and eigenvalues
        self.eigenvalues = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]
        
        print(f"PCA: Keeping {self.n_components} components")
        print(f"Variance explained: {np.sum(self.eigenvalues) / np.sum(eigenvalues):.4f}")
        
    def transform(self, X):
        """
        Transform data to PCA space.
        
        Args:
            X: Data matrix (N x D)
        
        Returns:
            Transformed data (N x n_components)
        """
        if self.mean is None or self.components is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Data matrix (N x D)
        
        Returns:
            Transformed data (N x n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def compute_covariance_inverse(self):
        """
        Compute inverse covariance matrix for Mahalanobis distance.
        Uses only the principal components for regularization.
        
        Returns:
            Inverse covariance matrix in reduced space
        """
        if self.components is None or self.eigenvalues is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        # Compute inverse covariance in reduced (PCA) space
        # In PCA space, covariance is diagonal with eigenvalues on diagonal
        # So inverse is just 1/eigenvalues on diagonal
        lambda_inv = np.diag(1.0 / (self.eigenvalues + 1e-10))
        self.cov_inv = lambda_inv
        
        return self.cov_inv
    
    def get_mahalanobis_cov_inv(self):
        """
        Get inverse covariance matrix for Mahalanobis distance computation.
        
        Returns:
            Inverse covariance matrix
        """
        if self.cov_inv is None:
            self.compute_covariance_inverse()
        return self.cov_inv
    
    def save(self, filepath):
        """
        Save PCA model to file.
        
        Args:
            filepath: Path to save file
        """
        model_data = {
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'mean': self.mean,
            'components': self.components,
            'eigenvalues': self.eigenvalues,
            'cov_inv': self.cov_inv
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"PCA model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load PCA model from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_components = model_data['n_components']
        self.variance_threshold = model_data['variance_threshold']
        self.mean = model_data['mean']
        self.components = model_data['components']
        self.eigenvalues = model_data['eigenvalues']
        self.cov_inv = model_data.get('cov_inv', None)
        
        print(f"PCA model loaded from {filepath}")
        print(f"Components: {self.n_components}")


def apply_pca_to_descriptors(descriptors, n_components=None, variance_threshold=0.95, 
                             pca_model=None, fit_new=True):
    """
    Apply PCA to a set of descriptors.
    
    Args:
        descriptors: Matrix of descriptors (N x D)
        n_components: Number of components (None = auto)
        variance_threshold: Variance to preserve
        pca_model: Pre-fitted PCA model (None = fit new)
        fit_new: Whether to fit a new model (ignored if pca_model provided)
    
    Returns:
        Tuple of (transformed_descriptors, pca_model)
    """
    if pca_model is None:
        if fit_new:
            pca_model = PCAProjection(n_components, variance_threshold)
            transformed = pca_model.fit_transform(descriptors)
        else:
            raise ValueError("Must provide pca_model or set fit_new=True")
    else:
        transformed = pca_model.transform(descriptors)
    
    return transformed, pca_model
