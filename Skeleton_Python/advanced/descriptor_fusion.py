"""
Descriptor Fusion Module
Combines multiple descriptors to improve retrieval accuracy
"""

import numpy as np
from scipy.io import loadmat
import os


class DescriptorFusion:
    """
    Fuses multiple descriptors using various fusion strategies.
    
    Fusion strategies:
    1. Early fusion: Concatenate descriptors before distance computation
    2. Late fusion: Compute distances separately then combine (score-level fusion)
    3. Weighted fusion: Assign different weights to different descriptors
    """
    
    def __init__(self, descriptor_types, weights=None, fusion_method='late'):
        """
        Initialize descriptor fusion.
        
        Args:
            descriptor_types: List of descriptor type names (e.g., ['global_hist_8', 'texture_8orient'])
            weights: List of weights for each descriptor (default: equal weights)
            fusion_method: 'early' or 'late' fusion
        """
        self.descriptor_types = descriptor_types
        self.fusion_method = fusion_method
        
        if weights is None:
            # Equal weights by default
            self.weights = np.ones(len(descriptor_types)) / len(descriptor_types)
        else:
            self.weights = np.array(weights)
            # Normalize weights
            self.weights = self.weights / np.sum(self.weights)
        
        self.descriptors_data = {}
        self.fused_descriptors = None
        
    def load_descriptors(self, descriptor_base_path='./descriptors'):
        """
        Load all descriptor types from disk.
        
        Args:
            descriptor_base_path: Base path where descriptors are stored
        """
        print(f"Loading {len(self.descriptor_types)} descriptor types...")
        
        for desc_type in self.descriptor_types:
            desc_path = os.path.join(descriptor_base_path, desc_type)
            if not os.path.exists(desc_path):
                raise FileNotFoundError(f"Descriptor path not found: {desc_path}")
            desc_files = sorted([f for f in os.listdir(desc_path) if f.endswith('.mat')])
            descriptors = []
            filenames = []
            for desc_file in desc_files:
                filepath = os.path.join(desc_path, desc_file)
                data = loadmat(filepath)
                # For BoVW+SIFT, descriptor type is 'bovw_sift', shape is already 1D
                if desc_type == 'bovw_sift':
                    descriptors.append(data['F'].flatten())
                else:
                    descriptors.append(data['F'].flatten())
                filenames.append(desc_file.replace('.mat', ''))
            self.descriptors_data[desc_type] = {
                'descriptors': np.array(descriptors),
                'filenames': filenames
            }
            print(f"  Loaded {desc_type}: {len(descriptors)} descriptors, dim={descriptors[0].shape[0]}")
        
        # Verify all descriptor types have the same images
        reference_filenames = self.descriptors_data[self.descriptor_types[0]]['filenames']
        for desc_type in self.descriptor_types[1:]:
            if self.descriptors_data[desc_type]['filenames'] != reference_filenames:
                raise ValueError(f"Descriptor {desc_type} has different images than reference")
        
        print(f"All descriptors loaded successfully!")
        
    def create_early_fusion(self):
        """
        Create early fusion by concatenating descriptors.
        
        Returns:
            Fused descriptor matrix (N x D_total)
        """
        if not self.descriptors_data:
            raise ValueError("No descriptors loaded. Call load_descriptors() first.")
        
        print("\nCreating early fusion (concatenation)...")
        
        # Normalize each descriptor type to unit norm
        normalized_descriptors = []
        
        for i, desc_type in enumerate(self.descriptor_types):
            descriptors = self.descriptors_data[desc_type]['descriptors']
            
            # L2 normalize each descriptor
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            normalized = descriptors / norms
            
            # Apply weight
            weighted = normalized * np.sqrt(self.weights[i])
            
            normalized_descriptors.append(weighted)
            print(f"  {desc_type}: dim={weighted.shape[1]}, weight={self.weights[i]:.3f}")
        
        # Concatenate all descriptors
        self.fused_descriptors = np.concatenate(normalized_descriptors, axis=1)
        
        print(f"Fused descriptor shape: {self.fused_descriptors.shape}")
        print(f"Total dimensions: {self.fused_descriptors.shape[1]}")
        
        return self.fused_descriptors
    
    def get_distance_matrices(self, distance_metric='euclidean'):
        """
        Compute distance matrices for each descriptor type (for late fusion).
        
        Args:
            distance_metric: Distance metric to use
            
        Returns:
            List of distance matrices, one per descriptor type
        """
        from cvpr_compare import batch_compare
        
        if not self.descriptors_data:
            raise ValueError("No descriptors loaded. Call load_descriptors() first.")
        
        print(f"\nComputing distance matrices with {distance_metric}...")
        
        n_images = len(self.descriptors_data[self.descriptor_types[0]]['filenames'])
        distance_matrices = []
        
        for desc_type in self.descriptor_types:
            descriptors = self.descriptors_data[desc_type]['descriptors']
            
            # Compute pairwise distances
            dist_matrix = np.zeros((n_images, n_images))
            
            for i in range(n_images):
                distances = batch_compare(descriptors[i], descriptors, distance_metric)
                dist_matrix[i] = distances
            
            distance_matrices.append(dist_matrix)
            print(f"  {desc_type}: distance matrix computed")
        
        return distance_matrices
    
    def late_fusion_distances(self, query_idx, distance_matrices):
        """
        Perform late fusion by combining distance scores.
        
        Args:
            query_idx: Index of the query image
            distance_matrices: List of distance matrices from get_distance_matrices()
            
        Returns:
            Fused distance array
        """
        # Normalize each distance vector to [0, 1]
        fused_distances = np.zeros(distance_matrices[0].shape[1])
        
        for i, dist_matrix in enumerate(distance_matrices):
            distances = dist_matrix[query_idx]
            
            # Min-max normalization
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > min_dist:
                normalized = (distances - min_dist) / (max_dist - min_dist)
            else:
                normalized = distances
            
            # Apply weight and accumulate
            fused_distances += self.weights[i] * normalized
        
        return fused_distances
    
    def get_fused_descriptors(self):
        """Get the fused descriptors (for early fusion)."""
        return self.fused_descriptors
    
    def get_filenames(self):
        """Get the list of image filenames."""
        return self.descriptors_data[self.descriptor_types[0]]['filenames']
    
    def save_fused_descriptors(self, output_path='./descriptors/fused'):
        """
        Save fused descriptors to disk.
        
        Args:
            output_path: Directory to save fused descriptors
        """
        if self.fused_descriptors is None:
            raise ValueError("No fused descriptors. Call create_early_fusion() first.")
        
        os.makedirs(output_path, exist_ok=True)
        
        filenames = self.get_filenames()
        
        for i, filename in enumerate(filenames):
            output_file = os.path.join(output_path, f"{filename}.mat")
            
            # Save in MATLAB format
            from scipy.io import savemat
            savemat(output_file, {'F': self.fused_descriptors[i]})
        
        print(f"\nFused descriptors saved to: {output_path}")
        print(f"Total files: {len(filenames)}")


def get_optimal_fusion_configs():
    """
    Return optimal fusion configurations based on complementary descriptors.
    
    Returns:
        List of fusion configurations
    """
    configs = [
        {
            'name': 'ColorTexture',
            'descriptors': ['global_hist_8', 'texture_12orient'],
            'weights': [0.6, 0.4],  # Color usually more discriminative
            'description': 'Global color + texture orientation'
        },
        {
            'name': 'GridTexture',
            'descriptors': ['grid2x2_color', 'texture_12orient'],
            'weights': [0.6, 0.4],
            'description': 'Spatial color grid + texture'
        },
        {
            'name': 'MultiScale',
            'descriptors': ['global_hist_8', 'grid4x4_color'],
            'weights': [0.5, 0.5],
            'description': 'Global + local color'
        },
        {
            'name': 'ColorTextureGrid',
            'descriptors': ['global_hist_8', 'texture_12orient', 'grid2x2_color'],
            'weights': [0.4, 0.3, 0.3],
            'description': 'Global color + texture + spatial grid'
        },
        {
            'name': 'FullFusion',
            'descriptors': ['global_hist_8', 'grid4x4_color', 'texture_12orient'],
            'weights': [0.4, 0.35, 0.25],
            'description': 'Comprehensive fusion of best descriptors'
        }
    ]
    
    return configs


if __name__ == "__main__":
    print("="*70)
    print("DESCRIPTOR FUSION MODULE")
    print("="*70)
    
    # Example: Create ColorTexture fusion
    print("\nExample: ColorTexture Fusion")
    
    fusion = DescriptorFusion(
        descriptor_types=['global_hist_8', 'texture_12orient'],
        weights=[0.6, 0.4],
        fusion_method='early'
    )
    
    fusion.load_descriptors()
    fused = fusion.create_early_fusion()
    fusion.save_fused_descriptors('./descriptors/fused_colortexture')
    
    print("\nDescriptor fusion example completed!")
