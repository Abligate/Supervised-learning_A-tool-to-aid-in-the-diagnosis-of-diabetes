# -*- coding: utf-8 -*-
"""
Assignment 2 - Full Experiment Matrix (Single-file, reproducible, plot-rich)

CPU-only + Absolute reproducibility edition
Key changes:
- Force CPU-only (disable GPU)
- Enable TF deterministic ops (even though GPU is off)
- Limit threads to 1 (TF + BLAS/OMP/MKL) to avoid parallel floating-point nondeterminism
- Strong seed setting (python/numpy/tf/keras)
- For NN training: fixed train/val split (no validation_split) + shuffle=False
- Clear TF/Keras session before each model fit to avoid state carryover
- Unified k-scan range for ALL clustering tasks: K = 1..40
"""

# ==========================================================
# Reproducibility MUST be set BEFORE importing numpy / tensorflow
# ==========================================================
import os

# ----- CPU-only -----
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU completely

# ----- Determinism -----
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# ----- Single-thread to avoid nondeterminism from parallel reductions -----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

# --- Prevent matplotlib GUI blocking ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc
)

# --- Keras import (compatible) ---
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, Callback
    from tensorflow.keras.regularizers import l2
except Exception:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, Callback
    from keras.regularizers import l2


# ==========================================================
# Config
# ==========================================================
@dataclass
class CFG:
    data_path: str = "diabetes_dataset.csv"
    out_root: str = "outputs_a2_full"

    # train/test
    test_ratio: float = 0.1

    # Seeds
    split_seed: int = 42
    algo_seed: int = 42
    nn_seed: int = 42

    # Feature decisions
    include_country: bool = False
    drop_patient_id: bool = True

    # DR settings
    dr_dim: int = 12
    pca_variance: float = 0.95

    # k selection (UNIFIED)
    scan_k: bool = True
    k_fixed: int = 5

    # 🔥 unified k scan range
    k_min: int = 1
    k_max: int = 40

    # Clustering settings
    kmeans_n_init: int = 20

    gmm_covariance_type: str = "full"
    gmm_n_init: int = 10
    gmm_max_iter: int = 500
    gmm_tol: float = 1e-3
    gmm_reg_covar: float = 1e-6
    gmm_k_criterion: str = "bic"  # "bic" or "aic"

    # NN base settings
    nn_epochs: int = 100
    nn_batch: int = 16
    nn_lr: float = 5e-4
    nn_patience: int = 15
    nn_activation: str = "elu"
    nn_dropout: float = 0.3
    nn_l2: float = 1e-3
    nn_width: int = 128

    # Task4 / Task5 mini-grid search
    task4_do_grid: bool = True
    task4_dr_dims: List[int] = field(default_factory=lambda: [6, 12, 18])
    task4_nn_widths: List[int] = field(default_factory=lambda: [64, 128])
    task4_nn_l2s: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    task4_activations: List[str] = field(default_factory=lambda: ["elu", "relu", "tanh"])

    task5_do_grid: bool = True
    task5_nn_widths: List[int] = field(default_factory=lambda: [64, 128])
    task5_nn_l2s: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    task5_activations: List[str] = field(default_factory=lambda: ["elu", "relu", "tanh"])

    save_nn_predictions: bool = True

    # deterministic runtime controls
    deterministic: bool = True
    threads: int = 1  # keep 1 for absolute reproducibility


# ==========================================================
# Utilities
# ==========================================================
def log(msg: str):
    print(msg)
    sys.stdout.flush()


def configure_tf_runtime(cfg: CFG):
    if not TF_AVAILABLE:
        return
    # Extra guard: ensure no GPU is visible
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    if cfg.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    try:
        tf.config.threading.set_inter_op_parallelism_threads(int(cfg.threads))
        tf.config.threading.set_intra_op_parallelism_threads(int(cfg.threads))
    except Exception:
        pass


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass


def ensure_dirs(base: str) -> Dict[str, str]:
    fig = os.path.join(base, "figures")
    tab = os.path.join(base, "tables")
    logs = os.path.join(base, "logs")
    os.makedirs(fig, exist_ok=True)
    os.makedirs(tab, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    return {"base": base, "fig": fig, "tab": tab, "log": logs}


def safe_savefig(path: str, dpi: int = 200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_dr_dim(X: np.ndarray, cfg: CFG, method: str, dr_dim_override: Optional[int] = None) -> Optional[int]:
    dim = cfg.dr_dim if dr_dim_override is None else dr_dim_override
    if dim is None:
        return None
    n_samples, n_features = X.shape
    if method.upper() == "RP":
        max_allowed = max(1, n_features)
    else:
        max_allowed = max(1, min(n_samples, n_features))
    d_eff = int(min(dim, max_allowed))
    if d_eff < dim:
        log(f"[WARN] {method}: requested dr_dim={dim} but data allows at most {max_allowed}. Using d_eff={d_eff}.")
    return d_eff


def plot_cluster_sizes(labels: np.ndarray, out_path: str, title: str):
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]
    plt.figure(figsize=(8, 4))
    plt.bar([str(u) for u in uniq], counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(title)
    safe_savefig(out_path)


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if labels is None or len(np.unique(labels)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def plot_k_scan_line(k_df: pd.DataFrame, out_path: str, title: str, y_col: str, y_label: Optional[str] = None):
    plt.figure(figsize=(10, 4))
    plt.plot(k_df["k"], k_df[y_col], marker="o")
    plt.xlabel("k")
    plt.ylabel(y_label or y_col)
    plt.title(title)
    safe_savefig(out_path)


# ==========================================================
# K selection (UNIFIED k=1..40)
# ==========================================================
def choose_k_kmeans_by_silhouette_allow_k1(
    X: np.ndarray, k_min: int, k_max: int, seed: int, n_init: int
) -> Tuple[int, pd.DataFrame]:
    """
    Scan k in [k_min..k_max]. Allows k=1:
    - For k=1: silhouette/DBI/CH are NaN (undefined), inertia is valid.
    - best_k is chosen by max silhouette among k>=2.
    """
    rows = []
    best_k, best_s = None, -1e18

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(X)

        if len(np.unique(labels)) < 2:
            sil = np.nan
            dbi = np.nan
            ch = np.nan
        else:
            sil = float(silhouette_score(X, labels))
            dbi = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))

        rows.append({
            "k": int(k),
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
            "inertia": float(km.inertia_)
        })

        if np.isfinite(sil) and sil > best_s:
            best_s = sil
            best_k = int(k)

    if best_k is None:
        # if everything breaks, fall back
        best_k = 2 if k_max >= 2 else k_min

    return int(best_k), pd.DataFrame(rows)


def choose_k_gmm_by_bic_aic(X: np.ndarray, k_min: int, k_max: int, cfg: CFG) -> Tuple[int, pd.DataFrame]:
    rows = []
    best_k = None
    best_val = 1e99

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cfg.gmm_covariance_type,
            random_state=cfg.algo_seed,
            n_init=cfg.gmm_n_init,
            max_iter=cfg.gmm_max_iter,
            tol=cfg.gmm_tol,
            reg_covar=cfg.gmm_reg_covar
        )
        gmm.fit(X)
        aic = float(gmm.aic(X))
        bic = float(gmm.bic(X))
        ll = float(gmm.score(X))
        rows.append({"k": int(k), "AIC": aic, "BIC": bic, "avg_loglik": ll})

        crit = bic if cfg.gmm_k_criterion.lower() == "bic" else aic
        if np.isfinite(crit) and crit < best_val:
            best_val = crit
            best_k = int(k)

    if best_k is None:
        best_k = int(k_min)
    return int(best_k), pd.DataFrame(rows)


def plot_2d_scatter(Z2: np.ndarray, color: Optional[np.ndarray], out_path: str, title: str, xlabel="Dim1", ylabel="Dim2"):
    plt.figure(figsize=(8, 6))
    if color is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], s=10)
    else:
        plt.scatter(Z2[:, 0], Z2[:, 1], c=color, s=10)
        plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    safe_savefig(out_path)


def plot_pca_variance(pca: PCA, out_path: str):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, evr, marker="o", label="Explained variance ratio")
    plt.plot(x, cum, marker="s", label="Cumulative explained variance")
    plt.xlabel("PC")
    plt.ylabel("Variance ratio")
    plt.title("PCA Explained Variance (Scree + Cumulative)")
    plt.legend()
    safe_savefig(out_path)


def plot_distance_preservation(X: np.ndarray, Z: np.ndarray, out_path: str, title: str, sample_n: int = 400, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    Xs = X[idx]
    Zs = Z[idx]
    m = min(2000, len(idx) * 5)
    i = rng.randint(0, len(idx), size=m)
    j = rng.randint(0, len(idx), size=m)
    dx = np.linalg.norm(Xs[i] - Xs[j], axis=1)
    dz = np.linalg.norm(Zs[i] - Zs[j], axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dz, s=8)
    plt.xlabel("Distance in original space")
    plt.ylabel("Distance in reduced space")
    plt.title(title)
    safe_savefig(out_path)


def kurtosis_approx(x: np.ndarray) -> float:
    x = x.astype(float)
    x = x - x.mean()
    v = (x**2).mean()
    if v <= 1e-12:
        return 0.0
    return float((x**4).mean() / (v**2) - 3.0)


def plot_ica_kurtosis(Z: np.ndarray, out_path: str, title: str):
    ks = [kurtosis_approx(Z[:, i]) for i in range(Z.shape[1])]
    plt.figure(figsize=(10, 4))
    plt.bar([f"IC{i+1}" for i in range(len(ks))], ks)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Excess kurtosis (approx)")
    plt.title(title)
    safe_savefig(out_path)


# ==========================================================
# Data prep
# ==========================================================
def build_raw_feature_frame(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    X = df.copy()
    drop_cols = []
    if cfg.drop_patient_id and "Patient_ID" in X.columns:
        drop_cols.append("Patient_ID")
    if "Diagnosis" in X.columns:
        drop_cols.append("Diagnosis")
    if not cfg.include_country and "Country" in X.columns:
        drop_cols.append("Country")
    X = X.drop(columns=drop_cols, errors="ignore")

    if "Gender" in X.columns:
        X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1, "F": 0, "M": 1})
        X["Gender"] = pd.to_numeric(X["Gender"], errors="coerce")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X


def build_a1_supervised_frame_no_scale(df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, str], List[str]]:
    d = df.copy()

    d["Glucose_Insulin_Ratio"] = d["Blood_Glucose_Level"] / (d["Insulin_Level"] + 1e-6)
    d["High_Glucose"] = (d["Blood_Glucose_Level"] > 140).astype(int)
    d["High_Insulin"] = (d["Insulin_Level"] > 90).astype(int)

    le = LabelEncoder()
    y = le.fit_transform(d["Diagnosis"].astype(str)).astype(int)
    label_map = {int(i): str(cls) for i, cls in enumerate(le.classes_)}

    if "Gender" in d.columns:
        d["Gender"] = d["Gender"].map({"Male": 1, "Female": 0, "M": 1, "F": 0})

    if cfg.include_country and "Country" in d.columns:
        d = pd.get_dummies(d, columns=["Country"], drop_first=True, dtype=int)
    else:
        if "Country" in d.columns:
            d = d.drop(columns=["Country"])

    if cfg.drop_patient_id and "Patient_ID" in d.columns:
        d = d.drop(columns=["Patient_ID"])

    cont_cols = ["Age", "Blood_Glucose_Level", "Insulin_Level", "Glucose_Insulin_Ratio"]
    for c in cont_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d[cont_cols] = d[cont_cols].fillna(d[cont_cols].median(numeric_only=True))

    X = d.drop(columns=["Diagnosis"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))

    return X, y, label_map, cont_cols


def scale_continuous_train_only(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, cont_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xtr = X_train_df.copy()
    Xte = X_test_df.copy()

    cols = [c for c in cont_cols if c in Xtr.columns]
    Xtr[cols] = scaler.fit_transform(Xtr[cols].values)
    Xte[cols] = scaler.transform(Xte[cols].values)
    return Xtr.values, Xte.values, scaler


# ==========================================================
# Task 1
# ==========================================================
def task1_raw_clustering(df: pd.DataFrame, cfg: CFG) -> Dict[str, object]:
    out = ensure_dirs(os.path.join(cfg.out_root, "task1_raw_cluster"))
    log("\n[TASK 1] Raw dataset clustering only (KMeans + EM/GMM)")

    X_df = build_raw_feature_frame(df, cfg)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    if cfg.scan_k:
        best_k_km, km_kdf = choose_k_kmeans_by_silhouette_allow_k1(
            X, cfg.k_min, cfg.k_max, cfg.algo_seed, cfg.kmeans_n_init
        )
        km_scan_csv = os.path.join(out["tab"], f"k_scan_raw_space_kmeans_k{cfg.k_min}_{cfg.k_max}.csv")
        km_kdf.to_csv(km_scan_csv, index=False)

        plot_k_scan_line(
            km_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_kmeans_silhouette_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space KMeans k selection (Silhouette; k=1 is NaN), k={cfg.k_min}..{cfg.k_max}",
            y_col="silhouette"
        )
        plot_k_scan_line(
            km_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_kmeans_inertia_elbow_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space KMeans elbow (Inertia), k={cfg.k_min}..{cfg.k_max}",
            y_col="inertia",
            y_label="inertia"
        )
        plot_k_scan_line(
            km_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_kmeans_calinski_harabasz_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space KMeans (Calinski-Harabasz), k={cfg.k_min}..{cfg.k_max}",
            y_col="calinski_harabasz"
        )
        plot_k_scan_line(
            km_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_kmeans_davies_bouldin_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space KMeans (Davies-Bouldin), k={cfg.k_min}..{cfg.k_max}",
            y_col="davies_bouldin"
        )

        best_k_gmm, gmm_kdf = choose_k_gmm_by_bic_aic(X, cfg.k_min, cfg.k_max, cfg)
        gmm_scan_csv = os.path.join(out["tab"], f"k_scan_raw_space_gmm_k{cfg.k_min}_{cfg.k_max}.csv")
        gmm_kdf.to_csv(gmm_scan_csv, index=False)

        crit_col = "BIC" if cfg.gmm_k_criterion.lower() == "bic" else "AIC"
        plot_k_scan_line(
            gmm_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_gmm_{crit_col}_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space GMM k selection ({crit_col} lower better), k={cfg.k_min}..{cfg.k_max}",
            y_col=crit_col
        )
        plot_k_scan_line(
            gmm_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_gmm_BIC_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space GMM k selection (BIC), k={cfg.k_min}..{cfg.k_max}",
            y_col="BIC"
        )
        plot_k_scan_line(
            gmm_kdf,
            os.path.join(out["fig"], f"k_scan_raw_space_gmm_AIC_k{cfg.k_min}_{cfg.k_max}.png"),
            f"Raw-space GMM k selection (AIC), k={cfg.k_min}..{cfg.k_max}",
            y_col="AIC"
        )
    else:
        best_k_km = int(cfg.k_fixed)
        best_k_gmm = int(cfg.k_fixed)
        km_scan_csv = None
        gmm_scan_csv = None

    km = KMeans(n_clusters=int(best_k_km), random_state=cfg.algo_seed, n_init=cfg.kmeans_n_init)
    km_labels = km.fit_predict(X)

    gmm = GaussianMixture(
        n_components=int(best_k_gmm),
        covariance_type=cfg.gmm_covariance_type,
        random_state=cfg.algo_seed,
        n_init=cfg.gmm_n_init,
        max_iter=cfg.gmm_max_iter,
        tol=cfg.gmm_tol,
        reg_covar=cfg.gmm_reg_covar
    )
    gmm_labels = gmm.fit_predict(X)

    log(f"[TASK1] KMeans best_k_by_silhouette = {best_k_km} (scan k={cfg.k_min}..{cfg.k_max})")
    log(f"[TASK1] GMM best_k_by_{cfg.gmm_k_criterion.upper()} = {best_k_gmm} (scan k={cfg.k_min}..{cfg.k_max})")

    rows = []
    m_km = clustering_metrics(X, km_labels)
    rows.append({"method": "KMeans", "k": int(best_k_km), **m_km, "inertia": float(km.inertia_)})
    m_g = clustering_metrics(X, gmm_labels)
    rows.append({
        "method": "GMM(EM)", "k": int(best_k_gmm), **m_g,
        "AIC": float(gmm.aic(X)), "BIC": float(gmm.bic(X)), "avg_loglik": float(gmm.score(X))
    })
    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(out["tab"], "metrics_raw_space.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    plot_cluster_sizes(km_labels, os.path.join(out["fig"], f"cluster_sizes_kmeans_bestk_{best_k_km}.png"),
                       f"Cluster sizes (KMeans, raw space, k={best_k_km})")
    plot_cluster_sizes(gmm_labels, os.path.join(out["fig"], f"cluster_sizes_gmm_bestk_{best_k_gmm}.png"),
                       f"Cluster sizes (GMM/EM, raw space, k={best_k_gmm})")

    pca2 = PCA(n_components=2, random_state=cfg.algo_seed)
    Z2 = pca2.fit_transform(X)
    plot_2d_scatter(Z2, km_labels, os.path.join(out["fig"], "rawspace_pca2_kmeans.png"),
                    "Raw space visualized by PCA(2D) colored by KMeans", xlabel="PC1", ylabel="PC2")
    plot_2d_scatter(Z2, gmm_labels, os.path.join(out["fig"], "rawspace_pca2_gmm.png"),
                    "Raw space visualized by PCA(2D) colored by GMM(EM)", xlabel="PC1", ylabel="PC2")

    labeled = df.copy()
    labeled["cluster_raw_kmeans"] = km_labels
    labeled["cluster_raw_gmm"] = gmm_labels
    labels_csv = os.path.join(out["tab"], "raw_cluster_labels.csv")
    labeled.to_csv(labels_csv, index=False)

    write_json(os.path.join(out["log"], "config_used.json"), cfg.__dict__)
    summary = (
        "Task1 done.\n"
        f"KMeans best_k_by_silhouette = {best_k_km} (scan k={cfg.k_min}..{cfg.k_max})\n"
        f"GMM best_k_by_{cfg.gmm_k_criterion.upper()} = {best_k_gmm} (scan k={cfg.k_min}..{cfg.k_max})\n\n"
        f"Saved:\n"
        f"- metrics: {metrics_csv}\n"
        f"- labels : {labels_csv}\n"
        f"- km scan csv: {km_scan_csv}\n"
        f"- gmm scan csv: {gmm_scan_csv}\n\n"
        f"{metrics_df.to_string(index=False)}\n"
    )
    write_text(os.path.join(out["log"], "summary.txt"), summary)

    return {"k_kmeans": int(best_k_km), "k_gmm": int(best_k_gmm)}


# ==========================================================
# Task 2
# ==========================================================
def task2_dr_only(df: pd.DataFrame, cfg: CFG):
    out = ensure_dirs(os.path.join(cfg.out_root, "task2_dr_only"))
    log("\n[TASK 2] DR only on raw dataset (PCA / ICA / RP)")

    X_df = build_raw_feature_frame(df, cfg)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    d_pca = resolve_dr_dim(X, cfg, "PCA")
    d_ica = resolve_dr_dim(X, cfg, "ICA")
    d_rp  = resolve_dr_dim(X, cfg, "RP")

    pca = PCA(n_components=d_pca, random_state=cfg.algo_seed) if d_pca is not None else PCA(
        n_components=cfg.pca_variance, random_state=cfg.algo_seed
    )
    Zp = pca.fit_transform(X)
    pd.DataFrame(Zp, columns=[f"PC{i+1}" for i in range(Zp.shape[1])]).to_csv(
        os.path.join(out["tab"], "pca_scores.csv"), index=False
    )
    plot_pca_variance(pca, os.path.join(out["fig"], "pca_explained_variance.png"))
    if Zp.shape[1] >= 2:
        plot_2d_scatter(Zp[:, :2], None, os.path.join(out["fig"], "pca_scatter_pc1_pc2.png"),
                        "PCA only: PC1 vs PC2", xlabel="PC1", ylabel="PC2")

    ica = FastICA(n_components=d_ica, random_state=cfg.algo_seed, max_iter=1000)
    Zi = ica.fit_transform(X)
    pd.DataFrame(Zi, columns=[f"IC{i+1}" for i in range(Zi.shape[1])]).to_csv(
        os.path.join(out["tab"], "ica_scores.csv"), index=False
    )
    if Zi.shape[1] >= 2:
        plot_2d_scatter(Zi[:, :2], None, os.path.join(out["fig"], "ica_scatter_ic1_ic2.png"),
                        "ICA only: IC1 vs IC2", xlabel="IC1", ylabel="IC2")
    plot_ica_kurtosis(Zi, os.path.join(out["fig"], "ica_kurtosis.png"),
                      "ICA component kurtosis (higher abs => more non-Gaussian)")

    rp = GaussianRandomProjection(n_components=d_rp, random_state=cfg.algo_seed)
    Zr = rp.fit_transform(X)
    pd.DataFrame(Zr, columns=[f"RP{i+1}" for i in range(Zr.shape[1])]).to_csv(
        os.path.join(out["tab"], "rp_scores.csv"), index=False
    )
    if Zr.shape[1] >= 2:
        plot_2d_scatter(Zr[:, :2], None, os.path.join(out["fig"], "rp_scatter_rp1_rp2.png"),
                        "RP only: RP1 vs RP2", xlabel="RP1", ylabel="RP2")
    plot_distance_preservation(X, Zr, os.path.join(out["fig"], "rp_distance_preservation.png"),
                               "RP distance preservation (sampled pairs)",
                               sample_n=400, seed=cfg.algo_seed)

    # ICA vs RP reconstruction MSE scan
    n_samples, n_features = X.shape
    Kmax = int(min(40, n_features))
    ks = list(range(1, Kmax + 1))

    ica_err, rp_err = [], []
    for k in ks:
        try:
            ica_k = FastICA(n_components=k, random_state=cfg.algo_seed, max_iter=2000, tol=1e-3)
            Zi_k = ica_k.fit_transform(X)
            if hasattr(ica_k, "inverse_transform"):
                Xi_hat = ica_k.inverse_transform(Zi_k)
            else:
                Xi_hat = Zi_k @ ica_k.mixing_.T
                if hasattr(ica_k, "mean_") and ica_k.mean_ is not None:
                    Xi_hat = Xi_hat + ica_k.mean_
            ica_err.append(float(np.mean((X - Xi_hat) ** 2)))
        except Exception:
            ica_err.append(np.nan)

        rp_k = GaussianRandomProjection(n_components=k, random_state=cfg.algo_seed)
        Zr_k = rp_k.fit_transform(X)
        R = rp_k.components_
        Xr_hat = Zr_k @ np.linalg.pinv(R.T)
        rp_err.append(float(np.mean((X - Xr_hat) ** 2)))

    err_df = pd.DataFrame({"k": ks, "ica_mse": ica_err, "rp_mse": rp_err})
    err_csv = os.path.join(out["tab"], "task2_reconstruction_mse_ica_rp_k_scan.csv")
    err_df.to_csv(err_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(err_df["k"], err_df["ica_mse"], linestyle="-", marker="s", linewidth=3,
             label="ICA recon MSE", zorder=5)
    plt.plot(err_df["k"], err_df["rp_mse"], linestyle="-", marker="^", linewidth=3, alpha=0.85,
             label="RP recon MSE (pseudo-inverse approx)", zorder=2)
    plt.xlabel("n_components (k)")
    plt.ylabel("Reconstruction MSE (on scaled X)")
    plt.title("Task2: Reconstruction error vs k (ICA vs RP)")
    plt.legend()
    safe_savefig(os.path.join(out["fig"], "task2_reconstruction_mse_compare_ica_rp_SOLID.png"))

    write_json(os.path.join(out["log"], "config_used.json"), cfg.__dict__)
    write_text(
        os.path.join(out["log"], "summary.txt"),
        f"Task2 done. requested_dr_dim={getattr(cfg,'dr_dim',None)} | effective: PCA={Zp.shape[1]}, ICA={Zi.shape[1]}, RP={Zr.shape[1]}\n"
        f"Also saved ICA vs RP reconstruction MSE scan: {err_csv}\n"
    )


# ==========================================================
# Task 3
# ==========================================================
def run_cluster_on_embedding_with_k_selection(Z: np.ndarray, cfg: CFG, tag_prefix: str, out_dir: Dict[str, str]) -> Dict[str, object]:
    """
    For each embedding space, select k separately for KMeans vs GMM.
    Uses unified scan range cfg.k_min..cfg.k_max (k=1..40).
    """
    if cfg.scan_k:
        best_k_km, km_kdf = choose_k_kmeans_by_silhouette_allow_k1(Z, cfg.k_min, cfg.k_max, cfg.algo_seed, cfg.kmeans_n_init)
        km_kdf.to_csv(os.path.join(out_dir["tab"], f"{tag_prefix}_k_scan_kmeans_k{cfg.k_min}_{cfg.k_max}.csv"), index=False)

        plot_k_scan_line(km_kdf, os.path.join(out_dir["fig"], f"{tag_prefix}_k_scan_kmeans_silhouette.png"),
                         f"{tag_prefix}: KMeans k selection (Silhouette; k=1 NaN), k={cfg.k_min}..{cfg.k_max}", y_col="silhouette")
        plot_k_scan_line(km_kdf, os.path.join(out_dir["fig"], f"{tag_prefix}_k_scan_kmeans_inertia_elbow.png"),
                         f"{tag_prefix}: KMeans elbow (Inertia), k={cfg.k_min}..{cfg.k_max}", y_col="inertia", y_label="inertia")
        plot_k_scan_line(km_kdf, os.path.join(out_dir["fig"], f"{tag_prefix}_k_scan_kmeans_calinski_harabasz.png"),
                         f"{tag_prefix}: KMeans (Calinski-Harabasz), k={cfg.k_min}..{cfg.k_max}", y_col="calinski_harabasz")
        plot_k_scan_line(km_kdf, os.path.join(out_dir["fig"], f"{tag_prefix}_k_scan_kmeans_davies_bouldin.png"),
                         f"{tag_prefix}: KMeans (Davies-Bouldin), k={cfg.k_min}..{cfg.k_max}", y_col="davies_bouldin")

        best_k_gmm, gmm_kdf = choose_k_gmm_by_bic_aic(Z, cfg.k_min, cfg.k_max, cfg)
        gmm_kdf.to_csv(os.path.join(out_dir["tab"], f"{tag_prefix}_k_scan_gmm_k{cfg.k_min}_{cfg.k_max}.csv"), index=False)

        crit_col = "BIC" if cfg.gmm_k_criterion.lower() == "bic" else "AIC"
        plot_k_scan_line(gmm_kdf, os.path.join(out_dir["fig"], f"{tag_prefix}_k_scan_gmm_{crit_col}.png"),
                         f"{tag_prefix}: GMM k selection ({crit_col} lower better), k={cfg.k_min}..{cfg.k_max}", y_col=crit_col)
    else:
        best_k_km = int(cfg.k_fixed)
        best_k_gmm = int(cfg.k_fixed)

    km = KMeans(n_clusters=int(best_k_km), random_state=cfg.algo_seed, n_init=cfg.kmeans_n_init)
    km_labels = km.fit_predict(Z)

    gmm = GaussianMixture(
        n_components=int(best_k_gmm),
        covariance_type=cfg.gmm_covariance_type,
        random_state=cfg.algo_seed,
        n_init=cfg.gmm_n_init,
        max_iter=cfg.gmm_max_iter,
        tol=cfg.gmm_tol,
        reg_covar=cfg.gmm_reg_covar
    )
    gmm_labels = gmm.fit_predict(Z)

    return {
        "km": km, "km_labels": km_labels, "k_km": int(best_k_km),
        "gmm": gmm, "gmm_labels": gmm_labels, "k_gmm": int(best_k_gmm),
    }


def task3_dr_plus_cluster(df: pd.DataFrame, cfg: CFG):
    out = ensure_dirs(os.path.join(cfg.out_root, "task3_dr_plus_cluster"))
    log("\n[TASK 3] For each DR: run KMeans + EM/GMM (6 results total)")

    X_df = build_raw_feature_frame(df, cfg)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    def save_block(tag: str, Z: np.ndarray):
        block = ensure_dirs(os.path.join(out["base"], tag))
        res = run_cluster_on_embedding_with_k_selection(Z, cfg, tag, block)

        rows = []
        m1 = clustering_metrics(Z, res["km_labels"])
        rows.append({"method": "KMeans", "k": res["k_km"], **m1, "inertia": float(res["km"].inertia_)})
        m2 = clustering_metrics(Z, res["gmm_labels"])
        rows.append({
            "method": "GMM(EM)", "k": res["k_gmm"], **m2,
            "AIC": float(res["gmm"].aic(Z)), "BIC": float(res["gmm"].bic(Z)), "avg_loglik": float(res["gmm"].score(Z))
        })
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(os.path.join(block["tab"], "metrics_summary.csv"), index=False)

        plot_cluster_sizes(res["km_labels"], os.path.join(block["fig"], "cluster_sizes_kmeans.png"),
                           f"{tag}: cluster sizes (KMeans, k={res['k_km']})")
        plot_cluster_sizes(res["gmm_labels"], os.path.join(block["fig"], "cluster_sizes_gmm.png"),
                           f"{tag}: cluster sizes (GMM/EM, k={res['k_gmm']})")

        if Z.shape[1] >= 2:
            plot_2d_scatter(Z[:, :2], res["km_labels"], os.path.join(block["fig"], "scatter_kmeans.png"),
                            f"{tag}: 2D scatter colored by KMeans")
            plot_2d_scatter(Z[:, :2], res["gmm_labels"], os.path.join(block["fig"], "scatter_gmm.png"),
                            f"{tag}: 2D scatter colored by GMM(EM)")

        pd.DataFrame({"label_kmeans": res["km_labels"], "label_gmm": res["gmm_labels"]}).to_csv(
            os.path.join(block["tab"], "cluster_labels.csv"), index=False
        )

        write_text(os.path.join(block["log"], "summary.txt"),
                   f"{tag} done.\nKMeans k={res['k_km']} | GMM k={res['k_gmm']}\n\n{metrics_df.to_string(index=False)}\n")

    d_pca = resolve_dr_dim(X, cfg, "PCA")
    d_ica = resolve_dr_dim(X, cfg, "ICA")
    d_rp  = resolve_dr_dim(X, cfg, "RP")

    pca = PCA(n_components=d_pca, random_state=cfg.algo_seed) if d_pca is not None else PCA(n_components=cfg.pca_variance, random_state=cfg.algo_seed)
    Zp = pca.fit_transform(X)
    save_block("PCA_plus_cluster", Zp)

    ica = FastICA(n_components=d_ica, random_state=cfg.algo_seed, max_iter=1000)
    Zi = ica.fit_transform(X)
    save_block("ICA_plus_cluster", Zi)

    rp = GaussianRandomProjection(n_components=d_rp, random_state=cfg.algo_seed)
    Zr = rp.fit_transform(X)
    save_block("RP_plus_cluster", Zr)

    write_json(os.path.join(out["log"], "config_used.json"), cfg.__dict__)
    write_text(os.path.join(out["log"], "summary.txt"),
               f"Task3 done. requested_dr_dim={cfg.dr_dim} | effective: PCA={Zp.shape[1]}, ICA={Zi.shape[1]}, RP={Zr.shape[1]}\n"
               f"Unified k-scan range: {cfg.k_min}..{cfg.k_max}\n")


# ==========================================================
# NN Helpers
# ==========================================================
class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.t0)


def build_nn(input_dim: int, cfg: CFG, width: Optional[int] = None, l2_strength: Optional[float] = None, activation: Optional[str] = None) -> Sequential:
    w = int(width if width is not None else cfg.nn_width)
    reg = float(l2_strength if l2_strength is not None else cfg.nn_l2)
    act = str(activation if activation is not None else cfg.nn_activation)

    w2 = max(8, w // 2)
    w3 = max(8, w // 4)

    model = Sequential()
    model.add(Dense(w, activation=act, input_shape=(input_dim,), kernel_regularizer=l2(reg)))
    model.add(Dropout(cfg.nn_dropout))
    model.add(Dense(w2, activation=act, kernel_regularizer=l2(reg)))
    model.add(Dropout(cfg.nn_dropout))
    model.add(Dense(w3, activation=act))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=cfg.nn_lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_training_curves(history, out_dir: str):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.legend()
    safe_savefig(os.path.join(out_dir, "nn_loss.png"))

    plt.figure(figsize=(10, 4))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy")
    plt.legend()
    safe_savefig(os.path.join(out_dir, "nn_accuracy.png"))


def plot_confusion(cm: np.ndarray, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    safe_savefig(out_path)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(loc="lower right")
    safe_savefig(out_path)
    return float(roc_auc)


def eval_nn(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {"acc": float(acc), "f1": float(f1), "cm": cm, "y_prob": y_prob, "y_pred": y_pred}


def deterministic_train_val_split(X: np.ndarray, y: np.ndarray, seed: int, val_ratio: float = 0.2):
    return train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y)


# ==========================================================
# Task 4
# ==========================================================
def task4_nn_with_dr(df: pd.DataFrame, cfg: CFG):
    out = ensure_dirs(os.path.join(cfg.out_root, "task4_nn_with_dr"))
    log("\n[TASK 4] Apply PCA/ICA/RP to Assignment1 NN and compare (+ grid search incl. activation)")

    X_df, y, label_map, cont_cols = build_a1_supervised_frame_no_scale(df, cfg)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=cfg.test_ratio, random_state=cfg.split_seed, stratify=y
    )
    X_train, X_test, _ = scale_continuous_train_only(X_train_df, X_test_df, cont_cols)

    dr_dims = cfg.task4_dr_dims if cfg.task4_do_grid else [cfg.dr_dim]
    widths = cfg.task4_nn_widths if cfg.task4_do_grid else [cfg.nn_width]
    l2s = cfg.task4_nn_l2s if cfg.task4_do_grid else [cfg.nn_l2]
    acts = cfg.task4_activations if cfg.task4_do_grid else [cfg.nn_activation]

    results_rows = []

    def run_nn_once(tag: str, Z_train: np.ndarray, Z_test: np.ndarray, dr_method: str,
                    dr_dim: Optional[int], width: int, l2_strength: float, activation: str, save_curves: bool):
        act_tag = activation.replace("/", "-").replace(" ", "")
        block_name = f"{tag}__drdim{dr_dim}__act{act_tag}__w{width}__l2{l2_strength:g}"
        block = ensure_dirs(os.path.join(out["base"], block_name))

        set_all_seeds(cfg.nn_seed)
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()

        model = build_nn(Z_train.shape[1], cfg, width=width, l2_strength=l2_strength, activation=activation)
        es = EarlyStopping(monitor="val_loss", patience=cfg.nn_patience, restore_best_weights=True)
        th = TimeHistory()

        Z_tr, Z_val, y_tr, y_val = deterministic_train_val_split(Z_train, y_train, seed=cfg.nn_seed, val_ratio=0.2)

        hist = model.fit(
            Z_tr, y_tr,
            validation_data=(Z_val, y_val),
            epochs=cfg.nn_epochs,
            batch_size=cfg.nn_batch,
            callbacks=[es, th],
            shuffle=False,
            verbose=0
        )

        if save_curves:
            plot_training_curves(hist, block["fig"])

        met = eval_nn(model, Z_test, y_test)
        plot_confusion(met["cm"], os.path.join(block["fig"], "confusion_matrix.png"))
        roc_auc = plot_roc(y_test, met["y_prob"], os.path.join(block["fig"], "roc.png"))

        row = {
            "setting": tag,
            "dr_method": dr_method,
            "dr_dim": dr_dim,
            "activation": str(activation),
            "nn_width": int(width),
            "nn_l2": float(l2_strength),
            "acc": met["acc"],
            "f1": met["f1"],
            "auc": roc_auc,
            "train_time_sec": float(sum(th.times)),
            "epochs_ran": int(len(hist.history.get("loss", []))),
            "input_dim": int(Z_train.shape[1]),
        }
        pd.DataFrame([row]).to_csv(os.path.join(block["tab"], "metrics.csv"), index=False)

        if cfg.save_nn_predictions:
            pd.DataFrame({"y_true": y_test, "y_prob": met["y_prob"], "y_pred": met["y_pred"]}).to_csv(
                os.path.join(block["tab"], "predictions.csv"), index=False
            )

        results_rows.append(row)

    # Baseline
    for act in acts:
        for width in widths:
            for l2_strength in l2s:
                run_nn_once("NN_BASELINE", X_train, X_test, "NONE", None, width, l2_strength, act, True)

    # DR loops
    for d in dr_dims:
        # PCA
        d_eff = resolve_dr_dim(X_train, cfg, "PCA", dr_dim_override=d)
        pca = PCA(n_components=d_eff, random_state=cfg.algo_seed)
        Ztr = pca.fit_transform(X_train)
        Zte = pca.transform(X_test)
        plot_pca_variance(pca, os.path.join(out["fig"], f"pca_explained_variance_task4_d{d_eff}.png"))
        if Ztr.shape[1] >= 2:
            plot_2d_scatter(Ztr[:, :2], y_train, os.path.join(out["fig"], f"pca_scatter_train_task4_d{d_eff}.png"),
                            f"Task4: PCA train (2D) colored by label (dr_dim={d_eff})", xlabel="PC1", ylabel="PC2")
        for act in acts:
            for width in widths:
                for l2_strength in l2s:
                    run_nn_once("NN_with_PCA", Ztr, Zte, "PCA", int(d_eff), width, l2_strength, act, True)

        # ICA
        d_eff = resolve_dr_dim(X_train, cfg, "ICA", dr_dim_override=d)
        ica = FastICA(n_components=d_eff, random_state=cfg.algo_seed, max_iter=1000)
        Ztr = ica.fit_transform(X_train)
        Zte = ica.transform(X_test)
        plot_ica_kurtosis(Ztr, os.path.join(out["fig"], f"ica_kurtosis_task4_d{d_eff}.png"),
                          f"Task4: ICA kurtosis (train, dr_dim={d_eff})")
        if Ztr.shape[1] >= 2:
            plot_2d_scatter(Ztr[:, :2], y_train, os.path.join(out["fig"], f"ica_scatter_train_task4_d{d_eff}.png"),
                            f"Task4: ICA train (2D) colored by label (dr_dim={d_eff})", xlabel="IC1", ylabel="IC2")
        for act in acts:
            for width in widths:
                for l2_strength in l2s:
                    run_nn_once("NN_with_ICA", Ztr, Zte, "ICA", int(d_eff), width, l2_strength, act, True)

        # RP
        d_eff = resolve_dr_dim(X_train, cfg, "RP", dr_dim_override=d)
        rp = GaussianRandomProjection(n_components=d_eff, random_state=cfg.algo_seed)
        Ztr = rp.fit_transform(X_train)
        Zte = rp.transform(X_test)
        plot_distance_preservation(X_train, Ztr, os.path.join(out["fig"], f"rp_distance_preservation_task4_d{d_eff}.png"),
                                   f"Task4: RP distance preservation (train, dr_dim={d_eff})", sample_n=400, seed=cfg.algo_seed)
        if Ztr.shape[1] >= 2:
            plot_2d_scatter(Ztr[:, :2], y_train, os.path.join(out["fig"], f"rp_scatter_train_task4_d{d_eff}.png"),
                            f"Task4: RP train (2D) colored by label (dr_dim={d_eff})", xlabel="RP1", ylabel="RP2")
        for act in acts:
            for width in widths:
                for l2_strength in l2s:
                    run_nn_once("NN_with_RP", Ztr, Zte, "RP", int(d_eff), width, l2_strength, act, True)

    summary = pd.DataFrame(results_rows).sort_values(["acc", "auc", "f1"], ascending=[False, False, False])
    summary.to_csv(os.path.join(out["tab"], "task4_grid_summary.csv"), index=False)

    best_rows = []
    for key in ["NONE", "PCA", "ICA", "RP"]:
        sub = summary[summary["dr_method"] == key]
        if len(sub) > 0:
            best_rows.append(sub.iloc[0].to_dict())
    pd.DataFrame(best_rows).to_csv(os.path.join(out["tab"], "task4_best_per_method.csv"), index=False)

    write_json(os.path.join(out["log"], "config_used.json"), cfg.__dict__)
    write_text(os.path.join(out["log"], "summary.txt"),
               f"Task4 done.\nLabel mapping: {label_map}\n\nTop 10 runs:\n{summary.head(10).to_string(index=False)}\n")


# ==========================================================
# Task 5
# ==========================================================
def one_hot_clusters(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], k), dtype=float)
    out[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return out


def task5_nn_with_cluster_features(df: pd.DataFrame, cfg: CFG):
    out = ensure_dirs(os.path.join(cfg.out_root, "task5_nn_with_cluster_features"))
    log("\n[TASK 5] Add raw-space clustering results as new features, rerun NN (+ grid search incl. activation)")

    X_df, y, label_map, cont_cols = build_a1_supervised_frame_no_scale(df, cfg)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=cfg.test_ratio, random_state=cfg.split_seed, stratify=y
    )
    X_train_base, X_test_base, _ = scale_continuous_train_only(X_train_df, X_test_df, cont_cols)

    rawX_df = build_raw_feature_frame(df, cfg)
    rawX_train = rawX_df.loc[X_train_df.index].copy()
    rawX_test = rawX_df.loc[X_test_df.index].copy()

    scaler_raw = StandardScaler()
    rawX_train_s = scaler_raw.fit_transform(rawX_train.values)
    rawX_test_s = scaler_raw.transform(rawX_test.values)

    if cfg.scan_k:
        best_k_km, kdf_km = choose_k_kmeans_by_silhouette_allow_k1(rawX_train_s, cfg.k_min, cfg.k_max, cfg.algo_seed, cfg.kmeans_n_init)
        kdf_km.to_csv(os.path.join(out["tab"], f"k_scan_train_rawspace_task5_kmeans_k{cfg.k_min}_{cfg.k_max}.csv"), index=False)

        plot_k_scan_line(kdf_km, os.path.join(out["fig"], "k_scan_train_rawspace_task5_kmeans_silhouette.png"),
                         f"Task5: KMeans k scan on TRAIN raw space (silhouette; k=1 NaN), k={cfg.k_min}..{cfg.k_max}", y_col="silhouette")
        plot_k_scan_line(kdf_km, os.path.join(out["fig"], "k_scan_train_rawspace_task5_kmeans_inertia_elbow.png"),
                         f"Task5: KMeans elbow on TRAIN raw space (inertia), k={cfg.k_min}..{cfg.k_max}", y_col="inertia", y_label="inertia")
        plot_k_scan_line(kdf_km, os.path.join(out["fig"], "k_scan_train_rawspace_task5_kmeans_calinski_harabasz.png"),
                         f"Task5: KMeans (CH) on TRAIN raw space, k={cfg.k_min}..{cfg.k_max}", y_col="calinski_harabasz")
        plot_k_scan_line(kdf_km, os.path.join(out["fig"], "k_scan_train_rawspace_task5_kmeans_davies_bouldin.png"),
                         f"Task5: KMeans (DBI) on TRAIN raw space, k={cfg.k_min}..{cfg.k_max}", y_col="davies_bouldin")

        best_k_gmm, kdf_gmm = choose_k_gmm_by_bic_aic(rawX_train_s, cfg.k_min, cfg.k_max, cfg)
        kdf_gmm.to_csv(os.path.join(out["tab"], f"k_scan_train_rawspace_task5_gmm_k{cfg.k_min}_{cfg.k_max}.csv"), index=False)

        crit_col = "BIC" if cfg.gmm_k_criterion.lower() == "bic" else "AIC"
        plot_k_scan_line(kdf_gmm, os.path.join(out["fig"], f"k_scan_train_rawspace_task5_gmm_{crit_col}.png"),
                         f"Task5: GMM k scan on TRAIN raw space ({crit_col} lower better), k={cfg.k_min}..{cfg.k_max}", y_col=crit_col)
    else:
        best_k_km = cfg.k_fixed
        best_k_gmm = cfg.k_fixed

    km = KMeans(n_clusters=int(best_k_km), random_state=cfg.algo_seed, n_init=cfg.kmeans_n_init)
    km_train = km.fit_predict(rawX_train_s)
    km_test = km.predict(rawX_test_s)

    gmm = GaussianMixture(
        n_components=int(best_k_gmm),
        covariance_type=cfg.gmm_covariance_type,
        random_state=cfg.algo_seed,
        n_init=cfg.gmm_n_init,
        max_iter=cfg.gmm_max_iter,
        tol=cfg.gmm_tol,
        reg_covar=cfg.gmm_reg_covar
    )
    gmm.fit(rawX_train_s)
    gmm_train = gmm.predict(rawX_train_s)
    gmm_test = gmm.predict(rawX_test_s)

    km_tr_oh = one_hot_clusters(km_train, int(best_k_km))
    km_te_oh = one_hot_clusters(km_test, int(best_k_km))
    gmm_tr_oh = one_hot_clusters(gmm_train, int(best_k_gmm))
    gmm_te_oh = one_hot_clusters(gmm_test, int(best_k_gmm))

    X_train_km = np.hstack([X_train_base, km_tr_oh])
    X_test_km = np.hstack([X_test_base, km_te_oh])

    X_train_gmm = np.hstack([X_train_base, gmm_tr_oh])
    X_test_gmm = np.hstack([X_test_base, gmm_te_oh])

    X_train_both = np.hstack([X_train_base, km_tr_oh, gmm_tr_oh])
    X_test_both = np.hstack([X_test_base, km_te_oh, gmm_te_oh])

    widths = cfg.task5_nn_widths if cfg.task5_do_grid else [cfg.nn_width]
    l2s = cfg.task5_nn_l2s if cfg.task5_do_grid else [cfg.nn_l2]
    acts = cfg.task5_activations if cfg.task5_do_grid else [cfg.nn_activation]

    def run_nn_grid(tag: str, Xtr: np.ndarray, Xte: np.ndarray, extra_info: Dict[str, object]):
        rows = []
        for act in acts:
            act_tag = act.replace("/", "-").replace(" ", "")
            for width in widths:
                for l2_strength in l2s:
                    block_name = f"{tag}__act{act_tag}__w{width}__l2{l2_strength:g}"
                    block = ensure_dirs(os.path.join(out["base"], block_name))

                    set_all_seeds(cfg.nn_seed)
                    if TF_AVAILABLE:
                        tf.keras.backend.clear_session()

                    model = build_nn(Xtr.shape[1], cfg, width=width, l2_strength=l2_strength, activation=act)
                    es = EarlyStopping(monitor="val_loss", patience=cfg.nn_patience, restore_best_weights=True)
                    th = TimeHistory()

                    X_tr, X_val, y_tr, y_val = deterministic_train_val_split(Xtr, y_train, seed=cfg.nn_seed, val_ratio=0.2)

                    hist = model.fit(
                        X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=cfg.nn_epochs,
                        batch_size=cfg.nn_batch,
                        callbacks=[es, th],
                        shuffle=False,
                        verbose=0
                    )

                    plot_training_curves(hist, block["fig"])
                    met = eval_nn(model, Xte, y_test)
                    plot_confusion(met["cm"], os.path.join(block["fig"], "confusion_matrix.png"))
                    roc_auc = plot_roc(y_test, met["y_prob"], os.path.join(block["fig"], "roc.png"))

                    row = {
                        "setting": tag,
                        "activation": str(act),
                        "nn_width": int(width),
                        "nn_l2": float(l2_strength),
                        "acc": met["acc"],
                        "f1": met["f1"],
                        "auc": roc_auc,
                        "train_time_sec": float(sum(th.times)),
                        "epochs_ran": int(len(hist.history.get("loss", []))),
                        "input_dim": int(Xtr.shape[1]),
                        **extra_info
                    }
                    pd.DataFrame([row]).to_csv(os.path.join(block["tab"], "metrics.csv"), index=False)

                    if cfg.save_nn_predictions:
                        pd.DataFrame({"y_true": y_test, "y_prob": met["y_prob"], "y_pred": met["y_pred"]}).to_csv(
                            os.path.join(block["tab"], "predictions.csv"), index=False
                        )

                    rows.append(row)
        return rows

    all_rows = []
    all_rows += run_nn_grid("NN_BASELINE_A1_FEATURES", X_train_base, X_test_base, {"kmeans_k": None, "gmm_k": None})
    all_rows += run_nn_grid("NN_WITH_KMEANS_CLUSTER_FEATURES", X_train_km, X_test_km, {"kmeans_k": int(best_k_km), "gmm_k": None})
    all_rows += run_nn_grid("NN_WITH_GMM_CLUSTER_FEATURES", X_train_gmm, X_test_gmm, {"kmeans_k": None, "gmm_k": int(best_k_gmm)})
    all_rows += run_nn_grid("NN_WITH_KM_PLUS_GMM_CLUSTER_FEATURES", X_train_both, X_test_both, {"kmeans_k": int(best_k_km), "gmm_k": int(best_k_gmm)})

    plot_cluster_sizes(km_train, os.path.join(out["fig"], "cluster_sizes_kmeans_train_task5.png"),
                       f"Task5: TRAIN cluster sizes (KMeans, k={best_k_km})")
    plot_cluster_sizes(gmm_train, os.path.join(out["fig"], "cluster_sizes_gmm_train_task5.png"),
                       f"Task5: TRAIN cluster sizes (GMM/EM, k={best_k_gmm})")

    summary = pd.DataFrame(all_rows).sort_values(["acc", "auc", "f1"], ascending=[False, False, False])
    summary.to_csv(os.path.join(out["tab"], "task5_grid_summary.csv"), index=False)

    best_rows = []
    for setting in [
        "NN_BASELINE_A1_FEATURES",
        "NN_WITH_KMEANS_CLUSTER_FEATURES",
        "NN_WITH_GMM_CLUSTER_FEATURES",
        "NN_WITH_KM_PLUS_GMM_CLUSTER_FEATURES",
    ]:
        sub = summary[summary["setting"] == setting]
        if len(sub) > 0:
            best_rows.append(sub.iloc[0].to_dict())
    pd.DataFrame(best_rows).to_csv(os.path.join(out["tab"], "task5_best_per_setting.csv"), index=False)

    write_json(os.path.join(out["log"], "config_used.json"), cfg.__dict__)
    write_text(os.path.join(out["log"], "summary.txt"),
               f"Task5 done.\nLabel mapping: {label_map}\n"
               f"Unified k-scan range: {cfg.k_min}..{cfg.k_max}\n\nTop 10 runs:\n{summary.head(10).to_string(index=False)}\n")


# ==========================================================
# Main
# ==========================================================
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Assignment2 full experiment matrix (plot-rich, reproducible)")
    parser.add_argument("--data", type=str, default="diabetes_dataset.csv")
    parser.add_argument("--out", type=str, default="outputs_a2_full")

    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--include_country", action="store_true")
    parser.add_argument("--no_scan_k", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dr_dim", type=int, default=12)

    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--algo_seed", type=int, default=42)
    parser.add_argument("--nn_seed", type=int, default=42)

    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--nn_width", type=int, default=128)
    parser.add_argument("--nn_l2", type=float, default=1e-3)

    parser.add_argument("--gmm_k_criterion", type=str, default="bic", choices=["bic", "aic"])

    parser.add_argument("--task4_no_grid", action="store_true")
    parser.add_argument("--task4_dr_dims", type=str, default="6,12,18")
    parser.add_argument("--task4_nn_widths", type=str, default="64,128")
    parser.add_argument("--task4_nn_l2s", type=str, default="1e-4,1e-3,1e-2")
    parser.add_argument("--task4_activations", type=str, default="elu,relu,tanh")

    parser.add_argument("--task5_no_grid", action="store_true")
    parser.add_argument("--task5_nn_widths", type=str, default="64,128")
    parser.add_argument("--task5_nn_l2s", type=str, default="1e-4,1e-3,1e-2")
    parser.add_argument("--task5_activations", type=str, default="elu,relu,tanh")

    parser.add_argument("--no_save_predictions", action="store_true")

    args = parser.parse_args()

    cfg = CFG(
        data_path=args.data,
        out_root=args.out,
        test_ratio=float(args.test_ratio),
        include_country=bool(args.include_country),
        scan_k=not bool(args.no_scan_k),
        k_fixed=int(args.k),
        dr_dim=int(args.dr_dim),
        split_seed=int(args.split_seed),
        algo_seed=int(args.algo_seed),
        nn_seed=int(args.nn_seed),
        nn_activation=str(args.activation),
        nn_lr=float(args.lr),
        nn_dropout=float(args.dropout),
        nn_width=int(args.nn_width),
        nn_l2=float(args.nn_l2),
        gmm_k_criterion=str(args.gmm_k_criterion),

        task4_do_grid=not bool(args.task4_no_grid),
        task4_dr_dims=parse_int_list(args.task4_dr_dims),
        task4_nn_widths=parse_int_list(args.task4_nn_widths),
        task4_nn_l2s=parse_float_list(args.task4_nn_l2s),
        task4_activations=parse_str_list(args.task4_activations),

        task5_do_grid=not bool(args.task5_no_grid),
        task5_nn_widths=parse_int_list(args.task5_nn_widths),
        task5_nn_l2s=parse_float_list(args.task5_nn_l2s),
        task5_activations=parse_str_list(args.task5_activations),

        save_nn_predictions=not bool(args.no_save_predictions),

        deterministic=True,
        threads=1
    )

    # lock TF runtime behavior
    configure_tf_runtime(cfg)

    # set global seeds once at start
    set_all_seeds(cfg.split_seed)

    os.makedirs(cfg.out_root, exist_ok=True)

    log("===== Assignment2 Full Experiment Matrix START (CPU-only deterministic) =====")
    log(f"Data: {cfg.data_path}")
    log(f"Output root: {cfg.out_root}")
    log(f"Seeds: split={cfg.split_seed}, algo={cfg.algo_seed}, nn={cfg.nn_seed}")
    log(f"Threads: {cfg.threads} | TF deterministic: {cfg.deterministic} | CPU-only: True")
    log(f"Split: test_ratio={cfg.test_ratio} (train_ratio={1.0 - cfg.test_ratio})")
    log(f"DR dim (default): {cfg.dr_dim} | k-scan: {cfg.scan_k} | include_country: {cfg.include_country}")
    log(f"Unified k-scan range: k={cfg.k_min}..{cfg.k_max}")
    log(f"GMM k-criterion: {cfg.gmm_k_criterion}")
    log(f"Task4 grid: {cfg.task4_do_grid} | dr_dims={cfg.task4_dr_dims} | widths={cfg.task4_nn_widths} | l2s={cfg.task4_nn_l2s} | activations={cfg.task4_activations}")
    log(f"Task5 grid: {cfg.task5_do_grid} | widths={cfg.task5_nn_widths} | l2s={cfg.task5_nn_l2s} | activations={cfg.task5_activations}")
    log(f"Save NN predictions: {cfg.save_nn_predictions}")

    df = pd.read_csv(cfg.data_path)

    task1_raw_clustering(df, cfg)
    task2_dr_only(df, cfg)
    task3_dr_plus_cluster(df, cfg)
    task4_nn_with_dr(df, cfg)
    task5_nn_with_cluster_features(df, cfg)

    log("===== DONE. See outputs in: " + cfg.out_root)


if __name__ == "__main__":
    main()
