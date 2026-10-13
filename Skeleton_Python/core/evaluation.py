"""
Evaluation metrics for visual search system.
Implements precision-recall computation, confusion matrices, and PR curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def get_class_from_filename(filename):
    """
    Extract class label from MSRC filename format.
    Format: CLASS_INSTANCE_s.bmp
    
    Args:
        filename: Image filename
    
    Returns:
        Class ID (integer)
    """
    basename = os.path.basename(filename)
    class_id = int(basename.split('_')[0])
    return class_id


def compute_precision_recall(retrieved_files, query_file, ground_truth_func=None):
    """
    Compute precision and recall at each position in retrieved results.
    
    Args:
        retrieved_files: List of retrieved filenames in rank order
        query_file: Query filename
        ground_truth_func: Function to determine if two images are in same class
    
    Returns:
        Tuple of (precisions, recalls)
    """
    if ground_truth_func is None:
        ground_truth_func = lambda f1, f2: get_class_from_filename(f1) == get_class_from_filename(f2)
    
    query_class = get_class_from_filename(query_file)
    
    # Count total relevant images (same class as query)
    # For MSRC, count images in the same class
    total_relevant = sum(1 for f in retrieved_files if ground_truth_func(query_file, f))
    
    if total_relevant == 0:
        return np.array([]), np.array([])
    
    precisions = []
    recalls = []
    num_relevant = 0
    
    for i, retrieved_file in enumerate(retrieved_files):
        # Check if retrieved image is relevant
        if ground_truth_func(query_file, retrieved_file):
            num_relevant += 1
        
        # Compute precision and recall at this position
        precision = num_relevant / (i + 1)
        recall = num_relevant / total_relevant
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls)


def compute_average_precision(precisions, recalls):
    """
    Compute Average Precision (AP) from precision-recall curve.
    
    Args:
        precisions: Array of precision values
        recalls: Array of recall values
    
    Returns:
        Average precision value
    """
    if len(precisions) == 0:
        return 0.0
    
    # Use 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        # Find maximum precision for recall >= t
        precs = precisions[recalls >= t]
        if len(precs) > 0:
            ap += np.max(precs)
    
    return ap / 11.0


def compute_precision_at_k(retrieved_files, query_file, k=10, ground_truth_func=None):
    """
    Compute precision at top K results.
    
    Args:
        retrieved_files: List of retrieved filenames in rank order
        query_file: Query filename
        k: Number of top results to consider
        ground_truth_func: Function to determine if two images are in same class
    
    Returns:
        Precision@K value
    """
    if ground_truth_func is None:
        ground_truth_func = lambda f1, f2: get_class_from_filename(f1) == get_class_from_filename(f2)
    
    # Count relevant images in top K
    num_relevant = sum(1 for f in retrieved_files[:k] if ground_truth_func(query_file, f))
    
    return num_relevant / k


def compute_confusion_matrix(all_queries, all_retrievals, num_classes=20, top_k=1):
    """
    Compute confusion matrix for visual search results.
    
    Args:
        all_queries: List of query filenames
        all_retrievals: List of lists of retrieved filenames (one per query)
        num_classes: Number of classes
        top_k: Consider top K retrievals for class prediction
    
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for query_file, retrieved_files in zip(all_queries, all_retrievals):
        true_class = get_class_from_filename(query_file)
        
        # Majority vote from top K retrievals
        predicted_classes = [get_class_from_filename(f) for f in retrieved_files[:top_k]]
        predicted_class = max(set(predicted_classes), key=predicted_classes.count)
        
        confusion[true_class - 1, predicted_class - 1] += 1
    
    return confusion


def plot_precision_recall_curve(precisions, recalls, title="Precision-Recall Curve", 
                                save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        precisions: Array or list of precision values (or list of arrays for multiple curves)
        recalls: Array or list of recall values (or list of arrays for multiple curves)
        title: Plot title
        save_path: Path to save plot (None = display only)
    """
    plt.figure(figsize=(10, 8))
    
    # Handle single or multiple curves
    if isinstance(precisions, list) and isinstance(precisions[0], np.ndarray):
        # Multiple curves
        for i, (prec, rec) in enumerate(zip(precisions, recalls)):
            plt.plot(rec, prec, linewidth=2, label=f"Query {i+1}")
    else:
        # Single curve
        plt.plot(recalls, precisions, linewidth=2, marker='o', markersize=3)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    if isinstance(precisions, list) and len(precisions) > 1:
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(confusion_matrix, class_names=None, title="Confusion Matrix",
                         save_path=None, normalize=False):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save plot
        normalize: Whether to normalize by row (true class)
    """
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion_matrix = confusion_matrix.astype(np.float32) / row_sums
    
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    num_classes = confusion_matrix.shape[0]
    tick_marks = np.arange(num_classes)
    
    if class_names is None:
        class_names = [str(i+1) for i in range(num_classes)]
    
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            if normalize:
                text = f'{value:.2f}'
            else:
                text = f'{int(value)}'
            
            plt.text(j, i, text, ha="center", va="center",
                    color="white" if value > thresh else "black",
                    fontsize=6)
    
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_retrieval_system(query_results, output_dir='results'):
    """
    Comprehensive evaluation of retrieval system.
    
    Args:
        query_results: List of dictionaries with keys:
            - 'query_file': Query filename
            - 'retrieved_files': List of retrieved filenames in rank order
            - 'distances': List of distances
        output_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics for each query
    all_precisions = []
    all_recalls = []
    all_aps = []
    p_at_10_list = []
    p_at_20_list = []
    
    for result in query_results:
        query_file = result['query_file']
        retrieved_files = result['retrieved_files']
        
        # Compute PR for this query
        precs, recs = compute_precision_recall(retrieved_files, query_file)
        
        if len(precs) > 0:
            all_precisions.append(precs)
            all_recalls.append(recs)
            
            # Compute AP
            ap = compute_average_precision(precs, recs)
            all_aps.append(ap)
            
            # Compute P@10
            p_at_10 = compute_precision_at_k(retrieved_files, query_file, k=10)
            p_at_10_list.append(p_at_10)
            
            # Compute P@20
            p_at_20 = compute_precision_at_k(retrieved_files, query_file, k=20)
            p_at_20_list.append(p_at_20)
    
    # Compute mean Average Precision (mAP)
    mAP = np.mean(all_aps) if len(all_aps) > 0 else 0.0
    mean_p_at_10 = np.mean(p_at_10_list) if len(p_at_10_list) > 0 else 0.0
    mean_p_at_20 = np.mean(p_at_20_list) if len(p_at_20_list) > 0 else 0.0
    
    # Compute confusion matrix
    all_queries = [r['query_file'] for r in query_results]
    all_retrievals = [r['retrieved_files'] for r in query_results]
    confusion = compute_confusion_matrix(all_queries, all_retrievals)
    
    # Plot average PR curve
    if len(all_recalls) > 0:
        # Interpolate to common recall levels
        recall_levels = np.linspace(0, 1, 100)
        avg_precisions = np.zeros_like(recall_levels)
        
        for precs, recs in zip(all_precisions, all_recalls):
            # Interpolate precision at common recall levels
            interp_precs = np.interp(recall_levels, recs, precs, left=0, right=0)
            avg_precisions += interp_precs
        
        avg_precisions /= len(all_recalls)
        
        plot_precision_recall_curve(avg_precisions, recall_levels,
                                    title=f"Average PR Curve (mAP={mAP:.4f})",
                                    save_path=os.path.join(output_dir, 'pr_curve.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion, title="Confusion Matrix",
                         save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    plot_confusion_matrix(confusion, title="Normalized Confusion Matrix",
                         save_path=os.path.join(output_dir, 'confusion_matrix_normalized.png'),
                         normalize=True)
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")
        f.write(f"Mean Precision@10: {mean_p_at_10:.4f}\n")
        f.write(f"Mean Precision@20: {mean_p_at_20:.4f}\n")
        f.write(f"Number of queries: {len(query_results)}\n")
    
    print(f"\nEvaluation Results:")
    print(f"  mAP: {mAP:.4f}")
    print(f"  Mean P@10: {mean_p_at_10:.4f}")
    print(f"  Mean P@20: {mean_p_at_20:.4f}")
    
    return {
        'mAP': mAP,
        'mean_p_at_10': mean_p_at_10,
        'mean_p_at_20': mean_p_at_20,
        'confusion_matrix': confusion
    }
