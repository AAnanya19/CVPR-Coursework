"""
Compute and save image descriptors for the entire dataset.
Supports multiple descriptor types and configurations.
"""

import os
import sys
import numpy as np
import cv2
import scipy.io as sio
from descriptors import extract_descriptor
from bovw_sift_descriptor import BoVWSIFTDescriptor

DATASET_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MSRC_ObjCategImageDatabase_v2'))
OUT_FOLDER = os.path.abspath('./descriptors')

# Descriptor configurations to compute
DESCRIPTOR_CONFIGS = [
    # Global color histograms with different quantization levels
    {'type': 'global_hist', 'bins_per_channel': 4, 'name': 'global_hist_4'},
    {'type': 'global_hist', 'bins_per_channel': 8, 'name': 'global_hist_8'},
    {'type': 'global_hist', 'bins_per_channel': 16, 'name': 'global_hist_16'},
    
    # Spatial grid color descriptor
    {'type': 'grid_color', 'grid_size': 4, 'bins_per_channel': 4, 'name': 'grid4x4_color'},
    
    # Texture descriptor
    {'type': 'texture', 'num_orientations': 12, 'name': 'texture_12orient'},
    # BoVW SIFT descriptor
    {'type': 'bovw_sift', 'n_words': 100, 'sift_nfeatures': 200, 'name': 'bovw_sift'},
]


def compute_all_descriptors():
    images_folder = os.path.join(DATASET_FOLDER, 'Images')
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found at {images_folder}")
        return
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.bmp')])
    if len(image_files) == 0:
        print(f"Error: No BMP files found in {images_folder}")
        return
    print(f"Found {len(image_files)} images")
    for config in DESCRIPTOR_CONFIGS:
        descriptor_name = config['name']
        descriptor_type = config.pop('type')
        out_subfolder = os.path.join(OUT_FOLDER, descriptor_name)
        os.makedirs(out_subfolder, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Computing descriptors: {descriptor_name}")
        print(f"Type: {descriptor_type}")
        print(f"Parameters: {config}")
        print(f"{'='*60}")
        if descriptor_type == 'bovw_sift':
            img_paths = [os.path.join(images_folder, fname) for fname in image_files]
            bovw = BoVWSIFTDescriptor(n_words=config.get('n_words', 100), sift_nfeatures=config.get('sift_nfeatures', 200))
            print("Extracting SIFT features and building vocabulary...")
            histograms = bovw.fit_transform(img_paths)
            for idx, filename in enumerate(image_files):
                fout = os.path.join(out_subfolder, filename.replace('.bmp', '.mat'))
                sio.savemat(fout, {'F': histograms[idx]})
            print(f"Completed {descriptor_name}")
        else:
            for idx, filename in enumerate(image_files):
                if (idx + 1) % 50 == 0:
                    print(f"  Processing {idx+1}/{len(image_files)}: {filename}")
                img_path = os.path.join(images_folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Warning: Could not load {filename}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                gt_folder = os.path.join(DATASET_FOLDER, 'GroundTruth')
                gt_path = os.path.join(gt_folder, filename.replace('.bmp', '_GT.bmp'))
                seg_folder = os.path.join(DATASET_FOLDER, 'SegmentationsGTHighQuality')
                seg_path = os.path.join(seg_folder, filename.replace('.bmp', '_HQGT.bmp'))
                mask = None
                gt_mask = None
                seg_mask = None
                if os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_img is not None:
                        gt_mask = (gt_img > 0).astype(np.uint8)
                if os.path.exists(seg_path):
                    seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    if seg_img is not None:
                        seg_mask = (seg_img > 0).astype(np.uint8)
                if gt_mask is not None and seg_mask is not None:
                    mask = np.logical_or(gt_mask, seg_mask).astype(np.uint8)
                elif gt_mask is not None:
                    mask = gt_mask
                elif seg_mask is not None:
                    mask = seg_mask
                try:
                    F = extract_descriptor(img, descriptor_type, mask=mask, **config)
                    fout = os.path.join(out_subfolder, filename.replace('.bmp', '.mat'))
                    sio.savemat(fout, {'F': F})
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
            print(f"Completed {descriptor_name}")
        config['type'] = descriptor_type


if __name__ == '__main__':
    print("Starting descriptor computation...")
    print(f"Dataset folder: {DATASET_FOLDER}")
    print(f"Output folder: {OUT_FOLDER}")
    
    compute_all_descriptors()
    
    print("\n" + "="*60)
    print("All descriptors computed successfully!")
    print("="*60)


