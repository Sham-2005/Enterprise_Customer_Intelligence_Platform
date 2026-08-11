"""
AI Customer Segmentation Engine for ECIP.
Implements feature scaling, PCA dimensionality reduction, multi-algorithm model evaluation
(K-Means, Hierarchical, GMM, DBSCAN), auto-cluster selection via Silhouette scores, and metrics scoring.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("ECIP.SegmentationEngine")

class CustomerSegmentationEngine:
    """Enterprise AI Unsupervised Clustering Engine with multi-algorithm benchmark and PCA visualization."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.settings = Settings(config_path)
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=3)

    def run_segmentation(
        self, feature_store_df: pd.DataFrame, target_k: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes feature preprocessing, model comparison, optimal cluster selection, and assignment.

        Returns:
            Tuple[segmented_df, model_benchmark_df, segmentation_metadata]
        """
        logger.info("Starting AI Customer Segmentation Pipeline...")
        df = feature_store_df.copy()

        # Step 1: Select numeric clustering features
        feature_cols = [
            "total_spending", "total_orders", "avg_order_value",
            "recency_days", "historical_clv", "avg_review_score_given",
            "distinct_categories_count", "loyalty_score"
        ]
        available_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Selected {len(available_cols)} clustering features: {available_cols}")

        X = df[available_cols].fillna(df[available_cols].median())

        # Step 2: Feature Scaling
        X_scaled = self.scaler.fit_transform(X)

        # Step 3: PCA Dimensionality Reduction (2D and 3D)
        pca_coords = self.pca.fit_transform(X_scaled)
        df["pca_x"] = pca_coords[:, 0]
        df["pca_y"] = pca_coords[:, 1]
        df["pca_z"] = pca_coords[:, 2]

        # Step 4: Auto-select optimal K or use override
        if target_k is None:
            optimal_k, k_scores = self._find_optimal_k(X_scaled)
            logger.info(f"Auto-selected optimal cluster count k = {optimal_k} based on Silhouette maximization.")
        else:
            optimal_k = target_k
            logger.info(f"Using manual override target_k = {optimal_k}")

        # Step 5: Benchmark Algorithms (K-Means, Agglomerative, GMM, DBSCAN)
        benchmark_results, best_model, final_labels = self._benchmark_algorithms(X_scaled, optimal_k)

        # Step 6: Attach cluster labels
        df["cluster_id"] = final_labels
        df["cluster_name"] = "Cluster " + df["cluster_id"].astype(str)

        metadata = {
            "optimal_k": optimal_k,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "feature_columns": available_cols
        }

        logger.info("AI Customer Segmentation Engine pipeline complete.")
        return df, benchmark_results, metadata

    def _find_optimal_k(self, X_scaled: np.ndarray, min_k: int = 3, max_k: int = 8) -> Tuple[int, Dict[int, float]]:
        scores = {}
        n_samples = len(X_scaled)
        sample_size = min(5000, n_samples)

        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels = km.fit_predict(X_scaled)

            if sample_size < n_samples:
                indices = np.random.RandomState(42).choice(n_samples, sample_size, replace=False)
                score = silhouette_score(X_scaled[indices], labels[indices])
            else:
                score = silhouette_score(X_scaled, labels)
            scores[k] = score

        best_k = max(scores, key=scores.get)
        return best_k, scores

    def _benchmark_algorithms(
        self, X_scaled: np.ndarray, k: int
    ) -> Tuple[pd.DataFrame, Any, np.ndarray]:
        logger.info(f"Benchmarking clustering algorithms at k = {k}...")

        n_samples = len(X_scaled)
        sample_size = min(5000, n_samples)
        sample_indices = np.random.RandomState(42).choice(n_samples, sample_size, replace=False) if sample_size < n_samples else np.arange(n_samples)

        models = {
            "K-Means": KMeans(n_clusters=k, random_state=42, n_init=3),
            "Gaussian Mixture (GMM)": GaussianMixture(n_components=k, random_state=42),
            "DBSCAN": DBSCAN(eps=0.8, min_samples=5)
        }

        # Include AgglomerativeClustering only if sample size is manageable
        if n_samples <= 10000:
            models["Hierarchical (Agglomerative)"] = AgglomerativeClustering(n_clusters=k)

        rows = []
        best_silhouette = -1.0
        best_model = None
        best_labels = None

        for name, model in models.items():
            try:
                if name == "Gaussian Mixture (GMM)":
                    labels = model.fit_predict(X_scaled)
                else:
                    labels = model.fit(X_scaled).labels_

                # Exclude noise for DBSCAN if applicable
                n_clusters_found = len(set(labels) - {-1})
                if n_clusters_found < 2:
                    continue

                X_sub = X_scaled[sample_indices]
                labels_sub = labels[sample_indices]

                sil = silhouette_score(X_sub, labels_sub)
                db = davies_bouldin_score(X_sub, labels_sub)
                ch = calinski_harabasz_score(X_sub, labels_sub)

                rows.append({
                    "Algorithm": name,
                    "Clusters Found": n_clusters_found,
                    "Silhouette Score": round(sil, 4),
                    "Davies-Bouldin Index": round(db, 4),
                    "Calinski-Harabasz Index": round(ch, 2)
                })

                if sil > best_silhouette:
                    best_silhouette = sil
                    best_model = model
                    best_labels = labels

            except Exception as e:
                logger.warning(f"Clustering algorithm {name} failed: {e}")

        # Fallback to KMeans if no model produced valid clusters
        if best_labels is None:
            km = KMeans(n_clusters=k, random_state=42, n_init=3)
            best_labels = km.fit_predict(X_scaled)
            best_model = km

        benchmark_df = pd.DataFrame(rows).sort_values(by="Silhouette Score", ascending=False) if rows else pd.DataFrame()
        return benchmark_df, best_model, best_labels

