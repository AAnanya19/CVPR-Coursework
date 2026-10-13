# Visual Search System Coursework

This project implements a modular Content-Based Image Retrieval (CBIR) system for the MSRC Object Category Image Database. The experiment focuses on maximizing retrieval accuracy using mask-aware descriptors, descriptor fusion, and query expansion. The pipeline supports region-based extraction using segmentation masks, fusion of multiple descriptors, and evaluation with standard metrics and visualizations.

## Latest Updates (2025)
- To run the system, maintain the folder structure as described below.

## How to Run
1. **Install Requirements:**
    - Make sure you have Python 3.8+ and the required packages:
      `pip install -r Skeleton_Python/requirements.txt`

2. **Prepare the Dataset:**
    - Place the MSRC Object Category Image Database in `MSRC_ObjCategImageDatabase_v2/` (in the project root).
    - Ensure `Images/`, `GroundTruth/`, and `SegmentationsGTHighQuality/` folders are present inside it.

3. **Run the Visual Search System:**
    - Start the interactive menu:
      `python3 Skeleton_Python/start.py`
    - Follow the menu to compute descriptors, run experiments, and launch the GUI.

4. **Compute Descriptors:**
    - Select option 2 from the menu or run:
      `python3 Skeleton_Python/core/cvpr_computedescriptors.py`
    - Descriptor files will be generated in `Skeleton_Python/descriptors/`.

5. **Run Experiments:**
    - Select option 3 or 8 from the menu to run fusion accuracy evaluation or the complete pipeline.
    - Results will be saved in `Skeleton_Python/results/`.

6. **Visualizations (Optional):**
    - To plot PR curves or descriptor comparison graphs, use:
      `python3 Skeleton_Python/experiments/plot_final_pr_curve.py`
      `python3 Skeleton_Python/results/descriptor_comparison_graph.py`


## Experiment Details
- **Descriptor Extraction:** Computes color, texture, and BoVW SIFT descriptors for each image, optionally using segmentation masks for region-based extraction.
- **Fusion & Query Expansion:** Combines multiple descriptors using weighted fusion and applies query expansion to improve retrieval accuracy.
- **Evaluation:** Measures performance using mAP, Precision@10, Precision@20, and generates visualizations for analysis.
- **Reporting:** Outputs per-category metrics, PR curves, and comparison tables to support analysis and reporting.

## Reproducibility & Customization
- All scripts are modular and can be adapted for new descriptors or fusion strategies.
- Parameters for fusion and query expansion are configurable in the main experiment script.
- Results and visualizations are automatically saved for inclusion in reports.