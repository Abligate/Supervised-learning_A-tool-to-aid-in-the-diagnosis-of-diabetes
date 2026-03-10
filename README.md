README — BG6013 Assignment 2 (Full Experiment Matrix)
1. Project overview

This repository contains a single-file, fully reproducible experiment pipeline for BG6013 Assignment 2. It covers:

Task 1: Raw-space clustering (K-Means + GMM/EM) with unified k scan (k=1..40)

Task 2: Dimensionality reduction only (PCA / ICA / Random Projection)

Task 3: DR + clustering (PCA/ICA/RP each combined with K-Means + GMM)

Task 4: DR + Neural Network (grid search over DR dims + NN hyperparameters)

Task 5: Clustering features + Neural Network (grid search over NN hyperparameters)

The script enforces CPU-only + deterministic execution (fixed seeds, deterministic ops, single-thread runtime, etc.) for maximum reproducibility. 

main

2. Files

Place the following files in the same folder:

main.py (the full pipeline script)

diabetes_dataset.csv (dataset)

3. Environment requirements

Recommended:

Python 3.9+

CPU execution (GPU is explicitly disabled by the script)

Python packages required (typical versions from common ML stacks are fine):

numpy

pandas

scikit-learn

matplotlib

tensorflow (preferred) or keras (fallback import is supported)

4. Installation (example)

Create and activate a virtual environment, then install dependencies:

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -U pip
pip install numpy pandas scikit-learn matplotlib tensorflow


If TensorFlow installation is not available on your machine, installing keras may still work (the script includes a fallback import path), but TensorFlow is recommended.

5. How to run (default full pipeline)

Run the full experiment matrix with default settings:

python main.py --data diabetes_dataset.csv --out outputs_a2_full


By default, the script will:

use test_ratio=0.1 (90/10 split),

run k-scan for clustering (k=1..40),

run Task4 and Task5 grid searches,

save all figures/tables/logs into the output folder.

6. Output structure (where to find results)

All outputs are written to the output root you set via --out (default outputs_a2_full). Typical structure:

outputs_a2_full/task1_raw_cluster/

clustering metrics, k-scan curves (silhouette/DB/CH/inertia), cluster size plots, PCA-2D colored visualizations

outputs_a2_full/task2_dr_only/

PCA explained variance + 2D scatter, ICA kurtosis + scatter, RP distance-preservation plot, DR scores tables

outputs_a2_full/task3_dr_plus_cluster/

subfolders: PCA_plus_cluster/, ICA_plus_cluster/, RP_plus_cluster/

each contains k-scan plots, metrics summary, cluster label outputs, 2D scatter colored by cluster

outputs_a2_full/task4_nn_with_dr/

grid search summaries (task4_grid_summary.csv, task4_best_per_method.csv), training curves, ROC, confusion matrices

outputs_a2_full/task5_nn_with_cluster_features/

grid search summaries (task5_grid_summary.csv, task5_best_per_setting.csv), k-scan on train only, NN plots, etc.

Each task folder includes:

figures/ (PNG plots)

tables/ (CSV summaries)

logs/ (config + summary text)

7. Common run options (CLI arguments)

You can control major experimental settings via command line flags:

Data / output

--data: path to dataset CSV (default diabetes_dataset.csv)

--out: output root directory (default outputs_a2_full)

Train/test split

--test_ratio: test fraction (default 0.1)

Feature inclusion

--include_country: include one-hot country features for supervised tasks (off by default)

Clustering k selection

--no_scan_k: disable k-scan and use a fixed k

--k: fixed k when --no_scan_k is used (default 5)

Unified scan range is internally set to k=1..40 (deterministic). 

main

Dimensionality reduction

--dr_dim: default DR dimension (used where applicable)

Seeds

--split_seed, --algo_seed, --nn_seed: control data split / algorithms / NN training seeds

Neural network hyperparameters (baseline defaults; grid search will override ranges)

--activation, --lr, --dropout, --nn_width, --nn_l2

GMM selection criterion

--gmm_k_criterion: bic or aic

Task4 (DR + NN) grid search controls

--task4_no_grid: disable Task4 grid search

--task4_dr_dims: e.g. "6,12,18"

--task4_nn_widths: e.g. "64,128"

--task4_nn_l2s: e.g. "1e-4,1e-3,1e-2"

--task4_activations: e.g. "elu,relu,tanh"

Task5 (Cluster features + NN) grid search controls

--task5_no_grid: disable Task5 grid search

--task5_nn_widths, --task5_nn_l2s, --task5_activations

Prediction saving

--no_save_predictions: do not save per-sample NN predictions CSV

8. Example: faster run (disable grid search)

If the reviewer wants a quicker run:

python main.py --data diabetes_dataset.csv --out outputs_quick --task4_no_grid --task5_no_grid

9. Reproducibility notes

This script is designed for absolute reproducibility:

GPU disabled

deterministic ops enabled

single-thread runtime set for TF/BLAS/OMP/MKL

fixed seeds for python/numpy/tensorflow

deterministic train/val split and shuffle=False during NN training 

main

If results differ on another machine, the most common causes are:

different TensorFlow / NumPy / BLAS versions

OS-level math library differences

non-identical dependency versions

overleaf Link(read only)https://www.overleaf.com/read/bcyqvfxqybzv#710d76
