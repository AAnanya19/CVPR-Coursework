"""
Image descriptor extraction functions for visual search system
Implements various descriptors including:
- Global color histograms with different quantization levels
- Spatial grid descriptors (color and texture)
- Texture descriptors using edge orientation histograms
"""

import numpy as np
import cv2
from scipy import ndimage


def extract_global_color_histogram(img, bins_per_channel=8, mask=None):
    """
    Extract global RGB color histogram from image.
    
    Args:
        img: Normalized RGB image (values in [0,1])
        bins_per_channel: Number of bins per color channel (default: 8)
    
    Returns:
        Flattened histogram as 1D numpy array
    """
    # Quantize each channel
    quantized = np.floor(img * bins_per_channel).astype(np.int32)
    quantized = np.clip(quantized, 0, bins_per_channel - 1)
    # Create 3D histogram
    hist = np.zeros((bins_per_channel, bins_per_channel, bins_per_channel))
    # Count pixels in each bin
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask is not None and mask[i, j] == 0:
                continue
            r, g, b = quantized[i, j]
            hist[r, g, b] += 1
    # Normalize histogram
    total_pixels = img.shape[0] * img.shape[1] if mask is None else np.count_nonzero(mask)
    hist = hist / (total_pixels if total_pixels > 0 else 1)
    # Flatten to 1D array
    return hist.flatten()


def extract_grid_color_histogram(img, grid_size=4, bins_per_channel=4, mask=None):
    """
    Extract spatial grid of color histograms.
    
    Args:
        img: Normalized RGB image (values in [0,1])
        grid_size: Number of grid cells per dimension (e.g., 4 = 4x4 grid)
        bins_per_channel: Number of bins per color channel
    
    Returns:
        Concatenated histograms from all grid cells
    """
    h, w = img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    descriptors = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid_size - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid_size - 1 else w
            cell = img[y1:y2, x1:x2]
            cell_mask = mask[y1:y2, x1:x2] if mask is not None else None
            hist = extract_global_color_histogram(cell, bins_per_channel, mask=cell_mask)
            descriptors.append(hist)
    return np.concatenate(descriptors)


def extract_texture_descriptor(img, num_orientations=8, mask=None):
    """
    Extract texture descriptor using edge orientation histogram.
    
    Args:
        img: Normalized RGB image (values in [0,1])
        num_orientations: Number of orientation bins
    
    Returns:
        Edge orientation histogram
    """
    # Convert to grayscale
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    
    # Convert orientation from [-pi, pi] to [0, 2*pi]
    orientation = (orientation + np.pi) % (2 * np.pi)
    
    # Create histogram
    hist = np.zeros(num_orientations)
    bin_size = 2 * np.pi / num_orientations
    
    # Weight by magnitude and accumulate into bins
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask is not None and mask[i, j] == 0:
                continue
            if magnitude[i, j] > 10:  # Threshold for significant edges
                bin_idx = int(orientation[i, j] / bin_size)
                bin_idx = min(bin_idx, num_orientations - 1)
                hist[bin_idx] += magnitude[i, j]
    # Normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def extract_descriptor(img, descriptor_type='global_hist', **kwargs):
    """
    Main function to extract descriptors based on type.
    
    Args:
        img: Normalized RGB image (values in [0,1])
        descriptor_type: Type of descriptor to extract
        **kwargs: Additional parameters for descriptor
    
    Returns:
        Feature descriptor as 1D numpy array
    """
    mask = kwargs.get('mask', None)
    if descriptor_type == 'global_hist':
        bins = kwargs.get('bins_per_channel', 8)
        return extract_global_color_histogram(img, bins_per_channel=bins, mask=mask)
    elif descriptor_type == 'grid_color':
        grid_size = kwargs.get('grid_size', 4)
        bins = kwargs.get('bins_per_channel', 4)
        return extract_grid_color_histogram(img, grid_size, bins, mask=mask)
    elif descriptor_type == 'texture':
        num_orientations = kwargs.get('num_orientations', 8)
        return extract_texture_descriptor(img, num_orientations, mask=mask)
    else:
        raise ValueError(f"Unknown descriptor type: {descriptor_type}")
