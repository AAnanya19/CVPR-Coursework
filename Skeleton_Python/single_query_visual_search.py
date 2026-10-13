"""
Run a single visual search query and display top 20 results.
"""


import os
import sys
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import fusion+query expansion system
sys.path.append(os.path.join(os.path.dirname(__file__), 'advanced'))
from test_fusion_query_expansion import FusionQueryExpansion, compute_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(project_root, '../MSRC_ObjCategImageDatabase_v2/Images')
descriptor_dir = os.path.join(project_root, 'descriptors/global_hist_8')

# Load image filenames
image_files = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]



# Allow user to set N (number of top results)
import argparse
parser = argparse.ArgumentParser(description='Single query visual search')
parser.add_argument('--N', type=int, default=20, help='Number of top results to display and evaluate')
parser.add_argument('--query', type=str, default=None, help='Query image filename (e.g., 15_10_s)')
args = parser.parse_args()
N = args.N
query_filename = args.query


# Use Set 15 fusion+query expansion for accuracy
fusion_config = {
    'descriptor_types': ['global_hist_8', 'grid4x4_color', 'texture_12orient', 'bovw_sift'],
    'weights': [0.10, 0.10, 0.30, 0.50],
    'alpha': 1.2,
    'beta': 1.5,
    'gamma': 0.20,
    'k_feedback': 40,
    'distance_metric': 'chi_square',
    'use_pca': False
}
fusion_system = FusionQueryExpansion(
    descriptor_types=fusion_config['descriptor_types'],
    fusion_weights=fusion_config['weights'],
    alpha=fusion_config['alpha'],
    beta=fusion_config['beta'],
    gamma=fusion_config['gamma'],
    k_feedback=fusion_config['k_feedback']
)
fusion_system.load_data()

# Pick query image by filename if provided, else random
if query_filename is not None:
    if query_filename.endswith('.bmp'):
        query_filename = query_filename.replace('.bmp', '')
    if query_filename in fusion_system.filenames:
        query_idx = fusion_system.filenames.index(query_filename)
        query_image = fusion_system.filenames[query_idx]
        logger.info(f"Query image (by filename): {query_image}")
    else:
        logger.error(f"Query image '{query_filename}' not found in dataset!")
        exit(1)
else:
    query_idx = random.randint(0, len(fusion_system.classes)-1)
    query_image = fusion_system.filenames[query_idx]
    logger.info(f"Query image (random): {query_image}")


# Use fusion+query expansion to get distances
distances_arr = fusion_system.retrieve_with_fusion_and_expansion(query_idx, distance_metric=fusion_config['distance_metric'])
all_classes = fusion_system.classes
query_class = all_classes[query_idx]

# Get top N results
sorted_indices = np.argsort(distances_arr)
topN_indices = [i for i in sorted_indices if i != query_idx][:N]
topN = [(fusion_system.filenames[i], distances_arr[i], all_classes[i]) for i in topN_indices]




logger.info(f"Top {N} results:")
correct_count = 0
for rank, (img, dist, result_class) in enumerate(topN, 1):
    if result_class == query_class:
        correct_count += 1
    logger.info(f"{rank:2d}. {img} (distance={dist:.4f}) {'✔' if result_class == query_class else ''}")


# Visualize query and top N results
plt.figure(figsize=(2*N, 3))
plt.subplot(1, N+1, 1)
img_path = os.path.join(image_dir, query_image + '.bmp' if not query_image.endswith('.bmp') else query_image)
plt.imshow(Image.open(img_path))
plt.title('Query')
plt.axis('off')
for i, (img, _, _) in enumerate(topN):
    plt.subplot(1, N+1, i+2)
    img_vis_path = os.path.join(image_dir, img + '.bmp' if not img.endswith('.bmp') else img)
    plt.imshow(Image.open(img_vis_path))
    plt.title(str(i+1))
    plt.axis('off')
plt.tight_layout()
plt.show()


# Compute and display Precision@N using fusion+query expansion
precision_at_N = correct_count / N
logger.info(f"\nPrecision@{N} for query '{query_image}': {precision_at_N:.2%} ({correct_count}/{N} correct)")


# After Precision@N calculation
# Compute confusion matrix for top N results
num_categories = 20
conf_matrix = np.zeros((num_categories, num_categories), dtype=int)

# For a single query, fill confusion matrix for top N results
for rank, (img, dist, result_class) in enumerate(topN, 1):
    conf_matrix[query_class-1, result_class-1] += 1

# Plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i+1) for i in range(num_categories)],
            yticklabels=[str(i+1) for i in range(num_categories)])
plt.xlabel('Retrieved Category')
plt.ylabel('Query Category')
plt.title(f'Confusion Matrix for Top {N} Results (Single Query)')
plt.tight_layout()
plt.show()

# Category labels
category_labels = [
    'Grass', 'Trees', 'Building', 'Airplane', 'Cow', 'Face', 'Car', 'Bicycle',
    'Sheep', 'Flowers', 'Sign', 'Bird', 'Books', 'Chair', 'Cat', 'Dog',
    'Road', 'Water', 'Body', 'Boat'
]

# Compute confusion matrix for all queries
num_categories = 20
conf_matrix_all = np.zeros((num_categories, num_categories), dtype=int)

for query_idx in range(len(fusion_system.classes)):
    query_class = fusion_system.classes[query_idx]
    distances_arr = fusion_system.retrieve_with_fusion_and_expansion(query_idx, distance_metric=fusion_config['distance_metric'])
    sorted_indices = np.argsort(distances_arr)
    topN_indices = [i for i in sorted_indices if i != query_idx][:N]
    for i in topN_indices:
        result_class = fusion_system.classes[i]
        conf_matrix_all[query_class-1, result_class-1] += 1

# Plot confusion matrix for all queries
plt.figure(figsize=(12,10))
sns.heatmap(conf_matrix_all, annot=True, fmt='d', cmap='Blues',
            xticklabels=category_labels, yticklabels=category_labels)
plt.xlabel('Retrieved Category')
plt.ylabel('Query Category')
plt.title(f'Confusion Matrix for Top {N} Results (All Queries)')
plt.tight_layout()
plt.savefig('Skeleton_Python/results/confusion_matrix_all_queries.png')
plt.show()
