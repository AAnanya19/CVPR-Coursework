"""
Automated experiment runner for visual search coursework.
Runs all required experiments and generates comprehensive results.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy.io as sio
import time
from cvpr_compare import batch_compare
from core.pca_utils import PCAProjection
from core.evaluation import evaluate_retrieval_system, get_class_from_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


DESCRIPTOR_FOLDER = os.path.abspath('./descriptors')
DATASET_FOLDER = os.path.abspath('../MSRC_ObjCategImageDatabase_v2')
RESULTS_FOLDER = os.path.abspath('./results')


def load_descriptors(descriptor_type):
    """
    Load descriptors from disk.
    
    Args:
        descriptor_type: Name of descriptor subfolder
    
    Returns:
        Tuple of (descriptors_array, filenames_array)
    """
    descriptor_path = os.path.join(DESCRIPTOR_FOLDER, descriptor_type)
    
    if not os.path.exists(descriptor_path):
        print(f"Warning: Descriptor folder not found: {descriptor_path}")
        return None, None
    
    # Load all descriptors
    all_files = sorted([f for f in os.listdir(descriptor_path) if f.endswith('.mat')])
    
    if len(all_files) == 0:
        print(f"Warning: No descriptor files found in {descriptor_path}")
        return None, None
    
    descriptors_list = []
    filenames_list = []
    
    for mat_file in all_files:
        mat_path = os.path.join(descriptor_path, mat_file)
        data = sio.loadmat(mat_path)
        descriptors_list.append(data['F'].flatten())
        
        # Get corresponding image filename
        img_filename = mat_file.replace('.mat', '.bmp')
        filenames_list.append(img_filename)
    
    descriptors = np.array(descriptors_list)
    filenames = np.array(filenames_list)
    
    print(f"Loaded {len(descriptors)} descriptors from {descriptor_type}")
    print(f"Descriptor dimensionality: {descriptors.shape[1]}")
    
    return descriptors, filenames


def run_experiment(descriptors, filenames, distance_metric='euclidean', 
                   use_pca=False, pca_components=50, experiment_name='experiment',
                   num_queries=30):
    """
    Run a single experiment configuration.
    
    Args:
        descriptors: Descriptor matrix
        filenames: Filename array
        distance_metric: Distance metric to use
        use_pca: Whether to apply PCA
        pca_components: Number of PCA components
        experiment_name: Name for this experiment
        num_queries: Number of random queries to test
    
    Returns:
        Evaluation metrics dictionary
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: {experiment_name}")
    print(f"Distance metric: {distance_metric}")
    print(f"Use PCA: {use_pca}")
    if use_pca:
        print(f"PCA components: {pca_components}")
    print(f"{'='*70}")
    
    output_dir = os.path.join(RESULTS_FOLDER, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply PCA if requested
    pca_model = None
    working_descriptors = descriptors
    
    if use_pca:
        print("Applying PCA...")
        pca_model = PCAProjection(n_components=pca_components, variance_threshold=0.95)
        working_descriptors = pca_model.fit_transform(descriptors)
        pca_model.compute_covariance_inverse()
        print(f"Reduced dimensionality: {descriptors.shape[1]} -> {working_descriptors.shape[1]}")
    
    # Select random queries (one from each class for diversity)
    np.random.seed(42)  # For reproducibility
    query_indices = []
    
    # Get one random image from each class
    for class_id in range(1, 21):  # Classes 1-20
        class_indices = [i for i, f in enumerate(filenames) 
                        if get_class_from_filename(f) == class_id]
        if len(class_indices) > 0:
            query_indices.append(np.random.choice(class_indices))
    
    # Add more random queries if needed
    while len(query_indices) < num_queries:
        idx = np.random.randint(0, len(filenames))
        if idx not in query_indices:
            query_indices.append(idx)
    
    query_indices = query_indices[:num_queries]
    
    print(f"Testing with {len(query_indices)} queries...")
    
    # Run searches
    query_results = []
    start_time = time.time()
    
    for i, query_idx in enumerate(query_indices):
        if (i + 1) % 10 == 0:
            print(f"  Processing query {i+1}/{len(query_indices)}...")
        
        query = working_descriptors[query_idx]
        
        # Compute distances
        if distance_metric == 'mahalanobis' and use_pca:
            cov_inv = pca_model.get_mahalanobis_cov_inv()
            distances = batch_compare(query, working_descriptors, distance_metric, cov_inv)
        else:
            distances = batch_compare(query, working_descriptors, distance_metric)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        sorted_filenames = filenames[sorted_indices].tolist()
        sorted_distances = distances[sorted_indices].tolist()
        
        query_results.append({
            'query_file': filenames[query_idx],
            'retrieved_files': sorted_filenames,
            'distances': sorted_distances
        })
    
    search_time = time.time() - start_time
    avg_time_per_query = search_time / len(query_indices)
    
    print(f"Search completed in {search_time:.2f}s (avg {avg_time_per_query:.4f}s per query)")
    
    # Evaluate
    print("Computing evaluation metrics...")
    metrics = evaluate_retrieval_system(query_results, output_dir)
    
    # Save experiment configuration and results
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Distance Metric: {distance_metric}\n")
        f.write(f"Use PCA: {use_pca}\n")
        if use_pca:
            f.write(f"PCA Components: {pca_components}\n")
            f.write(f"Original Dimensionality: {descriptors.shape[1]}\n")
            f.write(f"Reduced Dimensionality: {working_descriptors.shape[1]}\n")
        f.write(f"Number of Queries: {len(query_indices)}\n")
        f.write(f"Total Search Time: {search_time:.2f}s\n")
        f.write(f"Avg Time per Query: {avg_time_per_query:.4f}s\n")
    
    return metrics


def run_all_experiments():
    all_results = []
    """
    Run all required experiments for the coursework.
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    all_results = []
    
    # ========================================================================
    # REQUIREMENT 1: Global Color Histogram with different quantization levels
    # ========================================================================
    print("\n" + "="*70)
    print("REQUIREMENT 1: Global Color Histogram")
    print("="*70)
    
    for bins in [4, 8, 16]:
        descriptor_type = f'global_hist_{bins}'
        descriptors, filenames = load_descriptors(descriptor_type)
        
        if descriptors is not None:
            # Euclidean distance (required)
            metrics = run_experiment(
                descriptors, filenames,
                distance_metric='euclidean',
                use_pca=False,
                experiment_name=f'global_hist_{bins}_euclidean'
            )
            all_results.append((f'Global Hist {bins}x{bins}x{bins} - Euclidean', metrics))
    
    # ========================================================================
    # REQUIREMENT 3: Spatial Grid (Color and Texture)
    # ========================================================================
    print("\n" + "="*70)
    print("REQUIREMENT 3: Spatial Grid Descriptors")
    print("="*70)
    
    # ========================================================================
    # REQUIREMENT 3: Spatial Grid (Color, Texture, and BoVW+SIFT Fusion)
    # ========================================================================
    print("\n" + "="*70)
    print("REQUIREMENT 3: Spatial Grid + BoVW+SIFT Fusion")
    print("="*70)

    # Load all descriptors for fusion
    desc_types = ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift']
    desc_data = []
    desc_filenames = None
    for desc in desc_types:
        d, f = load_descriptors(desc)
        if d is not None:
            desc_data.append(d)
            if desc_filenames is None:
                desc_filenames = f

    # Only run fusion if all descriptors are available
    if len(desc_data) == len(desc_types):
        # Concatenate all descriptors for fusion
        fusion_descriptors = np.concatenate(desc_data, axis=1)
        metrics = run_experiment(
            fusion_descriptors, desc_filenames,
            distance_metric='chi_square',
            use_pca=False,
            experiment_name='fusion_global_grid_texture_bovw_chisq'
        )
        all_results.append(('Fusion: Global+Grid+Texture+BoVW - Chi-square', metrics))
    else:
        print("Warning: Not all fusion descriptors available. Skipping fusion experiment.")
    
    # ========================================================================
    # REQUIREMENT 4: PCA and Mahalanobis Distance
    # ========================================================================
    print("\n" + "="*70)
    print("REQUIREMENT 4: PCA and Mahalanobis Distance")
    print("="*70)
    
    # Use best descriptor from previous experiments
    descriptors, filenames = load_descriptors('global_hist_8')
    
    if descriptors is not None:
        # PCA with Euclidean
        for n_comp in [30, 50, 100]:
            metrics = run_experiment(
                descriptors, filenames,
                distance_metric='euclidean',
                use_pca=True,
                pca_components=n_comp,
                experiment_name=f'global_hist_8_pca{n_comp}_euclidean'
            )
            all_results.append((f'Global Hist 8 PCA-{n_comp} - Euclidean', metrics))
        
        # PCA with Mahalanobis
        for n_comp in [30, 50]:
            metrics = run_experiment(
                descriptors, filenames,
                distance_metric='mahalanobis',
                use_pca=True,
                pca_components=n_comp,
                experiment_name=f'global_hist_8_pca{n_comp}_mahalanobis'
            )
            all_results.append((f'Global Hist 8 PCA-{n_comp} - Mahalanobis', metrics))
    
    # ========================================================================
    # Generate Summary Report
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    summary_path = os.path.join(RESULTS_FOLDER, 'summary_report.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VISUAL SEARCH SYSTEM - EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Experiments: {len(all_results)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write(f"{'Experiment Name':<50} {'mAP':>8} {'P@10':>8} {'P@20':>8}\n")
        f.write("-"*80 + "\n")
        
        for name, metrics in all_results:
            mAP = metrics['mAP']
            p_at_10 = metrics['mean_p_at_10']
            p_at_20 = metrics['mean_p_at_20']
            f.write(f"{name:<50} {mAP:>8.4f} {p_at_10:>8.4f} {p_at_20:>8.4f}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Find best experiments
        best_map = max(all_results, key=lambda x: x[1]['mAP'])
        best_p10 = max(all_results, key=lambda x: x[1]['mean_p_at_10'])
        best_p20 = max(all_results, key=lambda x: x[1]['mean_p_at_20'])
        
        f.write("BEST RESULTS:\n")
        f.write(f"  Best mAP: {best_map[0]} (mAP={best_map[1]['mAP']:.4f})\n")
        f.write(f"  Best P@10: {best_p10[0]} (P@10={best_p10[1]['mean_p_at_10']:.4f})\n")
        f.write(f"  Best P@20: {best_p20[0]} (P@20={best_p20[1]['mean_p_at_20']:.4f})\n")
    
    print(f"\nSummary report saved to: {summary_path}")
    
    # Create comparison plot
    create_comparison_plot(all_results)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {RESULTS_FOLDER}")
    print("="*70)


def create_comparison_plot(all_results):
    """
    Create a comparison plot of all experiments.
    
    Args:
        all_results: List of (experiment_name, metrics) tuples
    """
    names = [r[0] for r in all_results]
    maps = [r[1]['mAP'] for r in all_results]
    p10s = [r[1]['mean_p_at_10'] for r in all_results]
    p20s = [r[1]['mean_p_at_20'] for r in all_results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    
    # mAP comparison
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, maps, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Mean Average Precision (mAP)', fontsize=12)
    ax1.set_title('Comparison of Experiments - mAP', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    
    # P@10 comparison
    ax2.barh(y_pos, p10s, color='coral')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Mean Precision@10', fontsize=12)
    ax2.set_title('Comparison of Experiments - P@10', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # P@20 comparison
    ax3.barh(y_pos, p20s, color='mediumseagreen')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('Mean Precision@20', fontsize=12)
    ax3.set_title('Comparison of Experiments - P@20', fontsize=14)
    ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_FOLDER, 'experiments_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("VISUAL SEARCH SYSTEM - AUTOMATED EXPERIMENT RUNNER")
    print("CVPR Coursework 2025")
    print("="*70)
    
    run_all_experiments()
