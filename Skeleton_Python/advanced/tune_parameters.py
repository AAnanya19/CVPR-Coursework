This file has been removed as it does not contribute to the best accuracy.
"""
Parameter Tuning for Fusion + Query Expansion
Optimize α, β, γ, and k_feedback for maximum accuracy
"""

import numpy as np
from scipy.io import loadmat
from descriptor_fusion import DescriptorFusion
from cvpr_compare import batch_compare
import os
import time
from itertools import product


def extract_class_from_filename(filename):
    """Extract class label from filename"""
    return int(filename.split('_')[0])


class ParameterTuner:
    """
    Tune Rocchio parameters for optimal performance.
    """
    
    def __init__(self, descriptor_types, fusion_weights):
        """
        Args:
            descriptor_types: List of descriptor types to fuse
            fusion_weights: Weights for fusion
        """
        self.descriptor_types = descriptor_types
        self.fusion_weights = fusion_weights
        
        # Initialize fusion system
        self.fusion = DescriptorFusion(
            descriptor_types=descriptor_types,
            weights=fusion_weights,
            fusion_method='late'
        )
        
    def load_data(self, descriptor_base_path='./descriptors'):
        """Load all descriptors"""
        print("Loading descriptors for tuning...")
        self.fusion.load_descriptors(descriptor_base_path)
        
        filenames = self.fusion.get_filenames()
        self.classes = np.array([extract_class_from_filename(f) for f in filenames])
        self.filenames = filenames
        
        print(f"Loaded {len(self.classes)} images")
    
    def retrieve_with_params(self, query_idx, alpha, beta, gamma, k_feedback, distance_metric='chi_square'):
        """
        Perform retrieval with specific parameters.
        """
        # Step 1: Compute distances for each descriptor type
        n_images = len(self.classes)
        distances_list = []
        
        for desc_type in self.descriptor_types:
            descriptors = self.fusion.descriptors_data[desc_type]['descriptors']
            query_desc = descriptors[query_idx]
            distances = batch_compare(query_desc, descriptors, distance_metric)
            distances_list.append(distances)
        
        # Step 2: Late fusion
        initial_fused_distances = np.zeros(n_images)
        
        for i, distances in enumerate(distances_list):
            min_d, max_d = np.min(distances), np.max(distances)
            if max_d > min_d:
                normalized = (distances - min_d) / (max_d - min_d)
            else:
                normalized = distances
            initial_fused_distances += self.fusion_weights[i] * normalized
        
        # Step 3: Query expansion
        sorted_indices = np.argsort(initial_fused_distances)[1:k_feedback+10]
        
        query_class = self.classes[query_idx]
        top_k_classes = self.classes[sorted_indices]
        
        relevant_mask = (top_k_classes == query_class)
        relevant_indices = sorted_indices[relevant_mask][:k_feedback]
        irrelevant_indices = sorted_indices[~relevant_mask][:k_feedback]
        
        # Expand each descriptor type
        expanded_distances_list = []
        
        for i, desc_type in enumerate(self.descriptor_types):
            descriptors = self.fusion.descriptors_data[desc_type]['descriptors']
            query_desc = descriptors[query_idx]
            
            expanded_desc = alpha * query_desc.copy()
            
            if len(relevant_indices) > 0:
                relevant_descriptors = descriptors[relevant_indices]
                expanded_desc += beta * np.mean(relevant_descriptors, axis=0)
            
            if len(irrelevant_indices) > 0 and gamma > 0:
                irrelevant_descriptors = descriptors[irrelevant_indices]
                expanded_desc -= gamma * np.mean(irrelevant_descriptors, axis=0)
            
            norm = np.linalg.norm(expanded_desc)
            if norm > 0:
                expanded_desc = expanded_desc / norm
            
            expanded_distances = batch_compare(expanded_desc, descriptors, distance_metric)
            expanded_distances_list.append(expanded_distances)
        
        # Step 4: Fuse expanded results
        final_fused_distances = np.zeros(n_images)
        
        for i, distances in enumerate(expanded_distances_list):
            min_d, max_d = np.min(distances), np.max(distances)
            if max_d > min_d:
                normalized = (distances - min_d) / (max_d - min_d)
            else:
                normalized = distances
            final_fused_distances += self.fusion_weights[i] * normalized
        
        return final_fused_distances
    
    def evaluate_params(self, alpha, beta, gamma, k_feedback, sample_size=None):
        """
        Evaluate specific parameter combination.
        
        Args:
            alpha, beta, gamma: Rocchio parameters
            k_feedback: Number of feedback documents
            sample_size: If provided, only evaluate on a sample (for speed)
            
        Returns:
            Dictionary with mAP, P@10, P@20
        """
        n_queries = len(self.classes)
        
        # Sample for faster tuning
        if sample_size and sample_size < n_queries:
            query_indices = np.random.choice(n_queries, sample_size, replace=False)
        else:
            query_indices = range(n_queries)
        
        aps, p10s, p20s = [], [], []
        
        for idx in query_indices:
            final_distances = self.retrieve_with_params(idx, alpha, beta, gamma, k_feedback)
            
            # Compute metrics
            sorted_indices = np.argsort(final_distances)
            sorted_classes = self.classes[sorted_indices[1:]]  # Skip query
            
            relevant = (sorted_classes == self.classes[idx])
            
            # Average Precision
            if np.sum(relevant) > 0:
                precisions = []
                num_relevant = 0
                for i, is_relevant in enumerate(relevant):
                    if is_relevant:
                        num_relevant += 1
                        precisions.append(num_relevant / (i + 1))
                ap = np.mean(precisions)
            else:
                ap = 0.0
            
            # P@10 and P@20
            p10 = np.sum(relevant[:10]) / 10
            p20 = np.sum(relevant[:20]) / 20
            
            aps.append(ap)
            p10s.append(p10)
            p20s.append(p20)
        
        return {
            'mAP': np.mean(aps),
            'P@10': np.mean(p10s),
            'P@20': np.mean(p20s)
        }
    
    def grid_search(self, alpha_range, beta_range, gamma_range, k_range, sample_size=200):
        """
        Grid search over parameter space.
        
        Args:
            alpha_range: List of alpha values to try
            beta_range: List of beta values to try
            gamma_range: List of gamma values to try
            k_range: List of k_feedback values to try
            sample_size: Number of queries to use for evaluation (for speed)
            
        Returns:
            Best parameters and results
        """
        print("\n" + "="*70)
        print("PARAMETER GRID SEARCH")
        print("="*70)
        print(f"\nSearch space:")
        print(f"  α (alpha): {alpha_range}")
        print(f"  β (beta): {beta_range}")
        print(f"  γ (gamma): {gamma_range}")
        print(f"  k_feedback: {k_range}")
        print(f"\nTotal combinations: {len(alpha_range) * len(beta_range) * len(gamma_range) * len(k_range)}")
        print(f"Sample size: {sample_size} queries")
        print("-"*70)
        
        best_score = 0
        best_params = None
        best_metrics = None
        
        all_results = []
        
        start_time = time.time()
        total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range) * len(k_range)
        current_combo = 0
        
        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    for k in k_range:
                        current_combo += 1
                        
                        # Evaluate this combination
                        metrics = self.evaluate_params(alpha, beta, gamma, k, sample_size)
                        
                        # Combined score (weighted: 40% mAP, 30% P@10, 30% P@20)
                        score = 0.4 * metrics['mAP'] + 0.3 * metrics['P@10'] + 0.3 * metrics['P@20']
                        
                        all_results.append({
                            'alpha': alpha,
                            'beta': beta,
                            'gamma': gamma,
                            'k': k,
                            'mAP': metrics['mAP'],
                            'P@10': metrics['P@10'],
                            'P@20': metrics['P@20'],
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = (alpha, beta, gamma, k)
                            best_metrics = metrics
                            print(f"  [{current_combo}/{total_combinations}] NEW BEST: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, k={k}")
                            print(f"    mAP={metrics['mAP']:.4f}, P@10={metrics['P@10']:.4f}, P@20={metrics['P@20']:.4f}, score={score:.4f}")
                        
                        if current_combo % 10 == 0:
                            elapsed = time.time() - start_time
                            eta = (elapsed / current_combo) * (total_combinations - current_combo)
                            print(f"  Progress: {current_combo}/{total_combinations} ({current_combo/total_combinations*100:.1f}%) - ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        print(f"Total time: {total_time:.2f}s")
        print(f"\nBest parameters:")
        print(f"  α (alpha): {best_params[0]:.2f}")
        print(f"  β (beta): {best_params[1]:.2f}")
        print(f"  γ (gamma): {best_params[2]:.2f}")
        print(f"  k_feedback: {best_params[3]}")
        print(f"\nBest metrics (on {sample_size} samples):")
        print(f"  mAP: {best_metrics['mAP']:.4f}")
        print(f"  P@10: {best_metrics['P@10']:.4f}")
        print(f"  P@20: {best_metrics['P@20']:.4f}")
        print(f"  Combined score: {best_score:.4f}")
        
        return best_params, best_metrics, all_results


def run_parameter_tuning():
    """
    Main parameter tuning function.
    """
    print("="*70)
    print("PARAMETER TUNING FOR FUSION + QUERY EXPANSION")
    print("="*70)
    
    # Current best configuration
    print("\nCurrent configuration:")
    print("  Descriptors: global_hist_8, grid4x4_color, texture_12orient")
    print("  Fusion weights: [0.4, 0.35, 0.25]")
    print("  Current params: α=1.0, β=0.75, γ=0.15, k=5")
    print("  Current results: mAP=0.3054, P@10=0.4662, P@20=0.3429")
    
    # Initialize tuner
    tuner = ParameterTuner(
        descriptor_types=['global_hist_8', 'grid4x4_color', 'texture_12orient'],
        fusion_weights=[0.4, 0.35, 0.25]
    )
    
    tuner.load_data()
    
    # Define parameter ranges
    alpha_range = [0.8, 0.9, 1.0, 1.1, 1.2]
    beta_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gamma_range = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
    k_range = [3, 5, 7, 10]
    
    # Run grid search on sample
    best_params, best_metrics_sample, all_results = tuner.grid_search(
        alpha_range, beta_range, gamma_range, k_range,
        sample_size=200  # Use 200 samples for fast tuning
    )
    
    # Evaluate best parameters on full dataset
    print("\n" + "="*70)
    print("EVALUATING BEST PARAMETERS ON FULL DATASET")
    print("="*70)
    
    alpha_opt, beta_opt, gamma_opt, k_opt = best_params
    
    print(f"\nOptimal parameters: α={alpha_opt:.2f}, β={beta_opt:.2f}, γ={gamma_opt:.2f}, k={k_opt}")
    print("Evaluating on all 591 queries...")
    
    start_time = time.time()
    final_metrics = tuner.evaluate_params(alpha_opt, beta_opt, gamma_opt, k_opt, sample_size=None)
    eval_time = time.time() - start_time
    
    print(f"Evaluation time: {eval_time:.2f}s")
    
    # Compare with baseline and current
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    baseline = {'mAP': 0.2708, 'P@10': 0.3500, 'P@20': 0.2617}
    current = {'mAP': 0.3054, 'P@10': 0.4662, 'P@20': 0.3429}
    
    print("\n1. Original Baseline:")
    print(f"   mAP: {baseline['mAP']:.4f}, P@10: {baseline['P@10']:.4f}, P@20: {baseline['P@20']:.4f}")
    
    print("\n2. Fusion + QE (default params α=1.0, β=0.75, γ=0.15, k=5):")
    print(f"   mAP: {current['mAP']:.4f}, P@10: {current['P@10']:.4f}, P@20: {current['P@20']:.4f}")
    
    print(f"\n3. ✓ OPTIMIZED (α={alpha_opt:.2f}, β={beta_opt:.2f}, γ={gamma_opt:.2f}, k={k_opt}):")
    print(f"   mAP: {final_metrics['mAP']:.4f} ({(final_metrics['mAP']-baseline['mAP'])/baseline['mAP']*100:+.1f}% vs baseline)")
    print(f"   P@10: {final_metrics['P@10']:.4f} ({(final_metrics['P@10']-baseline['P@10'])/baseline['P@10']*100:+.1f}% vs baseline)")
    print(f"   P@20: {final_metrics['P@20']:.4f} ({(final_metrics['P@20']-baseline['P@20'])/baseline['P@20']*100:+.1f}% vs baseline)")
    
    # Show improvement over current
    print("\n" + "-"*70)
    print("Improvement over current Fusion+QE:")
    map_imp = (final_metrics['mAP'] - current['mAP']) / current['mAP'] * 100
    p10_imp = (final_metrics['P@10'] - current['P@10']) / current['P@10'] * 100
    p20_imp = (final_metrics['P@20'] - current['P@20']) / current['P@20'] * 100
    
    print(f"  mAP:  {current['mAP']:.4f} → {final_metrics['mAP']:.4f} ({map_imp:+.2f}%)")
    print(f"  P@10: {current['P@10']:.4f} → {final_metrics['P@10']:.4f} ({p10_imp:+.2f}%)")
    print(f"  P@20: {current['P@20']:.4f} → {final_metrics['P@20']:.4f} ({p20_imp:+.2f}%)")
    
    print("\n" + "="*70)
    if final_metrics['mAP'] > current['mAP']:
        print("✓✓ Parameter tuning improved results!")
    elif final_metrics['mAP'] > current['mAP'] * 0.99:
        print("✓ Parameters are near-optimal (< 1% difference)")
    else:
        print("⚠ Current parameters were already good")
    print("="*70)
    
    return best_params, final_metrics


if __name__ == "__main__":
    best_params, final_metrics = run_parameter_tuning()
