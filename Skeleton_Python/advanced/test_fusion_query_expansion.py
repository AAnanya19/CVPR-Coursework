"""
Combined Fusion + Query Expansion System
Achieves maximum accuracy by combining both strategies
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.io import loadmat
from descriptor_fusion import DescriptorFusion
from core.cvpr_compare import batch_compare
import os
import time


def extract_class_from_filename(filename):
    """Extract class label from filename"""
    return int(filename.split('_')[0])


class FusionQueryExpansion:
    """
    Combines descriptor fusion with query expansion for maximum accuracy.
    """
    
    def __init__(self, descriptor_types, fusion_weights, 
                 alpha=1.0, beta=0.75, gamma=0.15, k_feedback=5):
        """
        Args:
            descriptor_types: List of descriptor types to fuse
            fusion_weights: Weights for fusion
            alpha, beta, gamma: Rocchio parameters for query expansion
            k_feedback: Number of feedback documents
        """
        self.descriptor_types = descriptor_types
        self.fusion_weights = fusion_weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_feedback = k_feedback
        
        # Initialize fusion system
        self.fusion = DescriptorFusion(
            descriptor_types=descriptor_types,
            weights=fusion_weights,
            fusion_method='late'
        )
    
    def load_data(self, descriptor_base_path='./descriptors'):
        """Load all descriptors"""
        print("Loading descriptors for fusion...")
        self.fusion.load_descriptors(descriptor_base_path)
        
        # Extract classes
        filenames = self.fusion.get_filenames()
        self.classes = np.array([extract_class_from_filename(f) for f in filenames])
        self.filenames = filenames
        
        print(f"Loaded {len(self.classes)} images")
    
    def retrieve_with_fusion_and_expansion(self, query_idx, distance_metric='chi_square'):
        """
        Perform retrieval using fusion + query expansion.
        
        Args:
            query_idx: Index of query image
            distance_metric: Distance metric to use
            
        Returns:
            Final fused and expanded distances
        """
        # Step 1: Compute distances for each descriptor type
        n_images = len(self.classes)
        distances_list = []
        
        for desc_type in self.descriptor_types:
            descriptors = self.fusion.descriptors_data[desc_type]['descriptors']
            query_desc = descriptors[query_idx]
            
            # Compute distances for this descriptor type
            distances = batch_compare(query_desc, descriptors, distance_metric)
            distances_list.append(distances)
        
        # Step 2: Late fusion to get initial ranking (manual fusion)
        initial_fused_distances = np.zeros(n_images)
        
        for i, distances in enumerate(distances_list):
            # Normalize distances to [0, 1]
            min_d, max_d = np.min(distances), np.max(distances)
            if max_d > min_d:
                normalized = (distances - min_d) / (max_d - min_d)
            else:
                normalized = distances
            
            initial_fused_distances += self.fusion_weights[i] * normalized
        
        # Step 3: Apply query expansion
        # Get top-k results from fused ranking
        sorted_indices = np.argsort(initial_fused_distances)[1:self.k_feedback+10]
        
        query_class = self.classes[query_idx]
        top_k_classes = self.classes[sorted_indices]
        
        # Identify relevant and irrelevant
        relevant_mask = (top_k_classes == query_class)
        relevant_indices = sorted_indices[relevant_mask][:self.k_feedback]
        irrelevant_indices = sorted_indices[~relevant_mask][:self.k_feedback]
        
        # Expand each descriptor type independently, then fuse again
        expanded_distances_list = []
        
        for i, desc_type in enumerate(self.descriptor_types):
            descriptors = self.fusion.descriptors_data[desc_type]['descriptors']
            query_desc = descriptors[query_idx]
            
            # Expand query for this descriptor type
            expanded_desc = self.alpha * query_desc.copy()
            
            if len(relevant_indices) > 0:
                relevant_descriptors = descriptors[relevant_indices]
                expanded_desc += self.beta * np.mean(relevant_descriptors, axis=0)
            
            if len(irrelevant_indices) > 0 and self.gamma > 0:
                irrelevant_descriptors = descriptors[irrelevant_indices]
                expanded_desc -= self.gamma * np.mean(irrelevant_descriptors, axis=0)
            
            # Normalize
            norm = np.linalg.norm(expanded_desc)
            if norm > 0:
                expanded_desc = expanded_desc / norm
            
            # Compute distances with expanded query
            expanded_distances = batch_compare(expanded_desc, descriptors, distance_metric)
            expanded_distances_list.append(expanded_distances)
        
        # Step 4: Fuse expanded results
        final_fused_distances = np.zeros(len(self.classes))
        
        for i, distances in enumerate(expanded_distances_list):
            # Normalize
            min_d, max_d = np.min(distances), np.max(distances)
            if max_d > min_d:
                normalized = (distances - min_d) / (max_d - min_d)
            else:
                normalized = distances
            
            final_fused_distances += self.fusion_weights[i] * normalized
        
        return final_fused_distances


def compute_metrics(distances, query_class, all_classes):
    """Compute AP, P@10, P@20"""
    sorted_indices = np.argsort(distances)
    sorted_classes = all_classes[sorted_indices[1:]]  # Skip query
    
    relevant = (sorted_classes == query_class)
    
    # Average Precision
    if np.sum(relevant) == 0:
        ap = 0.0
    else:
        precisions = []
        num_relevant = 0
        for i, is_relevant in enumerate(relevant):
            if is_relevant:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        ap = np.mean(precisions)
    
    # P@10 and P@20
    p10 = np.sum(relevant[:10]) / 10
    p20 = np.sum(relevant[:20]) / 20
    
    return ap, p10, p20


def evaluate_fusion_query_expansion():
    # Revert to last best P@20 value: Set 15 (Balanced, high feedback/gamma)
    configs = [
        {
            'name': 'Set 15 (Balanced, high feedback/gamma)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.10, 0.10, 0.30, 0.50],
            'alpha': 1.2,
            'beta': 1.5,
            'gamma': 0.20,
            'k_feedback': 40,
            'distance_metric': 'chi_square',
            'use_pca': False
        }
    ]
    """
    Evaluate the combined Fusion + Query Expansion system.
    """
    print("="*70)
    print("FUSION + QUERY EXPANSION EVALUATION")
    print("="*70)
    

    # List of parameter sets to try
    # Set 9 baseline
    configs = [
        {
            'name': 'Set 9 (More SIFT, higher beta)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.15, 0.10, 0.35, 0.40],
            'alpha': 1.2,
            'beta': 1.3,
            'gamma': 0.15,
            'k_feedback': 25,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        # Previous aggressive sets...
        {
            'name': 'Set 13 (Max SIFT, max feedback)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.05, 0.05, 0.20, 0.70],
            'alpha': 1.2,
            'beta': 1.7,
            'gamma': 0.20,
            'k_feedback': 50,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        {
            'name': 'Set 14 (Texture+SIFT, high beta/gamma)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.05, 0.05, 0.40, 0.50],
            'alpha': 1.2,
            'beta': 1.5,
            'gamma': 0.25,
            'k_feedback': 40,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        {
            'name': 'Set 15 (Balanced, high feedback/gamma)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.10, 0.10, 0.30, 0.50],
            'alpha': 1.2,
            'beta': 1.5,
            'gamma': 0.20,
            'k_feedback': 40,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        {
            'name': 'Set 16 (Texture focus, high beta/gamma)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.05, 0.05, 0.60, 0.30],
            'alpha': 1.2,
            'beta': 1.7,
            'gamma': 0.25,
            'k_feedback': 50,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        # Extreme sets for max P@20
        {
            'name': 'Set 17 (SIFT only, max feedback)',
            'descriptor_types': ['bovw_sift'],
            'weights': [1.0],
            'alpha': 1.2,
            'beta': 2.0,
            'gamma': 0.3,
            'k_feedback': 100,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        {
            'name': 'Set 18 (Texture+SIFT, max feedback)',
            'descriptor_types': ['texture_12orient', 'bovw_sift'],
            'weights': [0.2, 0.8],
            'alpha': 1.2,
            'beta': 2.0,
            'gamma': 0.3,
            'k_feedback': 80,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        {
            'name': 'Set 19 (Texture only, max feedback)',
            'descriptor_types': ['texture_12orient'],
            'weights': [1.0],
            'alpha': 1.2,
            'beta': 2.0,
            'gamma': 0.3,
            'k_feedback': 80,
            'distance_metric': 'chi_square',
            'use_pca': False
        },
        {
            'name': 'Set 20 (SIFT dominant, max feedback)',
            'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
            'weights': [0.01, 0.01, 0.08, 0.90],
            'alpha': 1.2,
            'beta': 2.0,
            'gamma': 0.3,
            'k_feedback': 60,
            'distance_metric': 'chi_square',
            'use_pca': False
        }
    ]


    all_results = []
    for config in configs:
        print("\nConfiguration:")
        print(f"  {config['name']}")
        print(f"  Descriptors: {', '.join(config['descriptor_types'])}")
        print(f"  Fusion weights: {config['weights']}")
        print(f"  Rocchio params: b1={config['alpha']}, b2={config['beta']}, b3={config['gamma']}")
        print(f"  k_feedback: {config['k_feedback']}")
        print(f"  Distance: {config['distance_metric']}")
        print("-"*70)

        system = FusionQueryExpansion(
            descriptor_types=config['descriptor_types'],
            fusion_weights=config['weights'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            k_feedback=config['k_feedback']
        )
        system.load_data()

        n_queries = len(system.classes)
        aps, p10s, p20s = [], [], []
        # For per-category metrics
        category_p20 = {}
        category_counts = {}
        category_names = {
            1: 'Grass', 2: 'Trees', 3: 'Building', 4: 'Airplane',
            5: 'Cow', 6: 'Face', 7: 'Car', 8: 'Bicycle',
            9: 'Sheep', 10: 'Flowers', 11: 'Sign', 12: 'Bird',
            13: 'Books', 14: 'Chair', 15: 'Cat', 16: 'Dog',
            17: 'Road', 18: 'Water', 19: 'Body', 20: 'Boat'
        }
        for cat in category_names:
            category_p20[cat] = []
            category_counts[cat] = 0

        start_time = time.time()
        for q_idx in range(n_queries):
            distances = system.retrieve_with_fusion_and_expansion(q_idx, distance_metric=config['distance_metric'])
            ap, p10, p20 = compute_metrics(distances, system.classes[q_idx], system.classes)
            aps.append(ap)
            p10s.append(p10)
            p20s.append(p20)
            cat = system.classes[q_idx]
            if cat in category_p20:
                category_p20[cat].append(p20)
                category_counts[cat] += 1
            if (q_idx+1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed/(q_idx+1))*(n_queries-q_idx-1)
                print(f"  Progress: {q_idx+1}/{n_queries} queries... Time: {elapsed:.1f}s, ETA: {eta:.1f}s")

        total_time = time.time() - start_time
        results = {
            'mAP': np.mean(aps),
            'P@10': np.mean(p10s),
            'P@20': np.mean(p20s),
            'time': total_time,
            'name': config['name'],
            'category_p20': category_p20,
            'category_counts': category_counts
        }
        all_results.append(results)

        print("RESULTS")
        print("="*70)
        print(f"mAP:  {results['mAP']:.4f}")
        print(f"P@10: {results['P@10']:.4f}")
        print(f"P@20: {results['P@20']:.4f}")
        print(f"Time: {results['time']:.2f}s ({results['time']/n_queries:.3f}s per query)")

        # Print per-category Precision@20 for Set 15 only
        if config['name'].startswith('Set 15'):
            print("\nPrecision@20 for each category:")
            print(f"{'Category':<12} {'Count':<6} {'P@20':<8}")
            for cat in sorted(category_names.keys()):
                cat_name = category_names[cat]
                count = results['category_counts'][cat]
                p20 = np.mean(results['category_p20'][cat]) if count > 0 else 0.0
                print(f"{cat_name:<12} {count:<6} {p20:.4f}")

    # Summary for all sets
    print("\n" + "="*70)
    print("SUMMARY OF ALL PARAMETER SETS")
    print("="*70)
    for res in all_results:
        print(f"{res['name']}: mAP={res['mAP']:.4f}, P@10={res['P@10']:.4f}, P@20={res['P@20']:.4f}")
    print("="*70)
    # Compare all sets to Set 9 (first config)
    baseline = all_results[0]
    print("\nComparison to Set 9 (More SIFT, higher beta):")
    for res in all_results[1:]:
        map_improvement = (res['mAP'] - baseline['mAP']) / baseline['mAP'] * 100
        p10_improvement = (res['P@10'] - baseline['P@10']) / baseline['P@10'] * 100
        p20_improvement = (res['P@20'] - baseline['P@20']) / baseline['P@20'] * 100
        print(f"{res['name']}: ΔmAP={map_improvement:+.2f}%, ΔP@10={p10_improvement:+.2f}%, ΔP@20={p20_improvement:+.2f}%")
    print("="*70)
    return all_results


if __name__ == "__main__":

    results = evaluate_fusion_query_expansion()
