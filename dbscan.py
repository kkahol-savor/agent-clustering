"""
This module provides functionality to find the best DBSCAN model for a given dataset
after reducing it to 2 PCA components. It includes a test function using the Wine dataset.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine
from sklearn.utils import Bunch


def optimize_dbscan_params(
    x_pca: np.ndarray,
    eps_range: np.ndarray,
    min_samples_range: range,
) -> Tuple[float, Dict[str, Any], Optional[np.ndarray]]:
    """
    Optimize DBSCAN parameters for the PCA-transformed data.

    Parameters:
        x_pca (np.ndarray): 2D PCA-transformed data.
        eps_range (np.ndarray): Array of epsilon values.
        min_samples_range (range): Range of min_samples values.

    Returns:
        Tuple containing:
        - best_score (float): Best silhouette score found.
        - best_params (dict): Best parameters as {'eps': value, 'min_samples': value}.
        - best_labels (np.ndarray or None): Cluster labels for best model.
    """
    best_score = -1.0
    best_params: Dict[str, Any] = {}
    best_labels: Optional[np.ndarray] = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            # pylint: disable=broad-exception-caught
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(x_pca)
                # Exclude noise (-1) when counting clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    score = silhouette_score(x_pca, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {"eps": eps, "min_samples": min_samples}
                        best_labels = labels
            except Exception:
                continue
            # pylint: enable=broad-exception-caught

    return best_score, best_params, best_labels


def find_best_dbscan_model(
    x: np.ndarray,
    eps_range: np.ndarray = np.linspace(0.1, 2.0, 20),
    min_samples_range: range = range(2, 20),
) -> Tuple[Optional[DBSCAN], Optional[np.ndarray], np.ndarray]:
    """
    Find the best DBSCAN model for the input data after reducing it to 2 PCA components.

    Parameters:
        x (np.ndarray): Input data.
        eps_range (np.ndarray): Range of epsilon values for DBSCAN (default: 0.1 to 2.0, 20 steps).
        min_samples_range (range): Range of min_samples values (default: 2 to 19).

    Returns:
        A tuple of:
        - best_dbscan (DBSCAN or None): Best DBSCAN model.
        - best_labels (np.ndarray or None): Cluster labels from the best model.
        - x_pca (np.ndarray): PCA-transformed data (2 components).
    """
    # Standardize the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Reduce to 2 components using PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    best_score, best_params, best_labels = optimize_dbscan_params(
        x_pca, eps_range, min_samples_range
    )

    best_dbscan: Optional[DBSCAN] = None
    if best_params:
        best_dbscan = DBSCAN(
            eps=best_params["eps"], min_samples=best_params["min_samples"]
        )
        best_dbscan.fit(x_pca)
        print(
            f"Best DBSCAN Parameters: eps={best_params['eps']:.2f}, "
            f"min_samples={best_params['min_samples']}"
        )
        print(f"Silhouette Score: {best_score:.4f}")
    else:
        print("No valid DBSCAN clustering found.")

    return best_dbscan, best_labels, x_pca


def test_with_wine_dataset() -> None:
    """
    Test the find_best_dbscan_model function using the Wine dataset.
    Loads the dataset, finds the best DBSCAN model, and visualizes the results.
    """
    wine_data: Bunch = load_wine()
    # Use attributes from the Bunch object; suppress Pylint no-member warnings
    x_data = wine_data.data  # pylint: disable=no-member
    y_data = wine_data.target  # pylint: disable=no-member

    best_dbscan, best_labels, x_pca = find_best_dbscan_model(x_data)

    if best_dbscan is not None and best_labels is not None:
        # Visualize the clustering results
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            x_pca[:, 0], x_pca[:, 1], c=best_labels, cmap="viridis"
        )
        plt.title("DBSCAN Clustering on Wine Dataset (2D PCA)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Cluster Labels")
        plt.show()

        # Visualize true labels for comparison
        plt.figure(figsize=(10, 6))
        scatter_true = plt.scatter(
            x_pca[:, 0], x_pca[:, 1], c=y_data, cmap="viridis"
        )
        plt.title("True Labels on Wine Dataset (2D PCA)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter_true, label="True Labels")
        plt.show()


if __name__ == "__main__":
    test_with_wine_dataset()
