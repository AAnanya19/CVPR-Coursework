# Content-Based Image Retrieval (CBIR) System

A modular Content-Based Image Retrieval system built for the **MSRC Object Category Image Database v2** (591 images, 20 classes). The system combines handcrafted visual descriptors, mask-aware feature extraction, weighted late fusion, and Rocchio query expansion to achieve strong retrieval performance using classical computer vision techniques.

---

## Results

| System Configuration | P@10 | P@20 | mAP |
|---|---|---|---|
| Global Colour Histogram (baseline) | 25.4% | 19.8% | 0.274 |
| + Chi-Square Distance | 35.0% | 26.2% | 0.271 |
| + Spatial Grid + Texture Fusion | 36.5% | 28.9% | 0.252 |
| + BoVW-SIFT + Late Fusion + Query Expansion | **69.19%** | **51.62%** | **0.451** |

> Final system achieves **~172% improvement** in P@10 over the Euclidean baseline.

---

## Features

- **Multiple Descriptors:** Global Colour Histogram, Spatial Grid (4×4), Edge Orientation Histogram, Bag-of-Visual-Words (BoVW) with SIFT
- **Mask-Aware Extraction:** Feature extraction restricted to segmentation ROI to suppress background noise
- **Distance Metrics:** Euclidean (L2), Chi-Square, Mahalanobis
- **Dimensionality Reduction:** PCA for compact feature representations
- **Fusion Strategy:** Weighted Score-Level (Late) Fusion across all descriptors
- **Query Expansion:** Rocchio algorithm for pseudo-relevance feedback
- **Evaluation:** mAP, Precision@10, Precision@20, PR curves, confusion matrices

---

## Project Structure

```
├── MSRC_ObjCategImageDatabase_v2/
│   ├── Images/
│   ├── GroundTruth/
│   └── SegmentationsGTHighQuality/
├── Skeleton_Python/
│   ├── core/
│   │   └── cvpr_computedescriptors.py
│   ├── experiments/
│   │   └── plot_final_pr_curve.py
│   ├── descriptors/
│   ├── results/
│   │   └── descriptor_comparison_graph.py
│   ├── start.py
│   └── requirements.txt
```

---

## Getting Started

### 1. Install Requirements

```bash
pip install -r Skeleton_Python/requirements.txt
```

### 2. Prepare the Dataset

Download the [MSRC Object Category Image Database v2](http://download.microsoft.com/download/3/3/9/339D8A24-47D7-412F-A1E8-1A415BC48A15/msrc_objcategimagedatabase_v2.zip) and place it in the project root:

```
MSRC_ObjCategImageDatabase_v2/
├── Images/
├── GroundTruth/
└── SegmentationsGTHighQuality/
```

### 3. Run the System

Launch the interactive menu:

```bash
python3 Skeleton_Python/start.py
```

### 4. Compute Descriptors

```bash
python3 Skeleton_Python/core/cvpr_computedescriptors.py
```

Descriptor files will be saved to `Skeleton_Python/descriptors/`.

### 5. Run Experiments

Select option **3** (fusion accuracy) or **8** (full pipeline) from the menu. Results are saved to `Skeleton_Python/results/`.

### 6. Visualisations (Optional)

```bash
# Precision-Recall curves
python3 Skeleton_Python/experiments/plot_final_pr_curve.py

# Descriptor comparison graphs
python3 Skeleton_Python/results/descriptor_comparison_graph.py
```

---

## Methodology

### Descriptor Pipeline

1. **Global Colour Histogram** — RGB colour distribution with 8 bins/channel (512D); Chi-Square distance
2. **Spatial Grid Descriptor** — 4×4 grid of local colour histograms to capture spatial layout
3. **Edge Orientation Histogram** — Sobel-based texture descriptor with 12 orientation bins
4. **BoVW-SIFT** — SIFT keypoints clustered into a 100-word visual vocabulary via k-Means

### Fusion & Query Expansion

- Descriptors are scored independently and combined via **weighted late fusion** (empirically tuned weights)
- **Rocchio Query Expansion** (α=0.80, β=1.00, γ=0.20, k=10) refines the query vector using top-ranked results as pseudo-relevant feedback

### Evaluation

Performance is measured using **mAP**, **P@10**, and **P@20** averaged across all 20 object categories. PR curves and confusion matrices are generated for qualitative analysis.

---

## Tech Stack

- **Language:** Python 3.8+
- **Libraries:** OpenCV, NumPy, SciPy, scikit-learn, Matplotlib
- **Dataset:** MSRC Object Category Image Database v2 (591 images, 20 classes)
