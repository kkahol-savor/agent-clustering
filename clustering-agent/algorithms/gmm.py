"""
Gaussian Mixture Models clustering implementation for the clustering agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

def run_gmm(data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Gaussian Mixture Model clustering on the provided data.
    
    Args:
        data: Pandas DataFrame with features
        hyperparameters: Dictionary of hyperparameters for GMM
        
    Returns:
        results: Dictionary with clustering results
    """
    # Extract hyperparameters with defaults
    n_components = hyperparameters.get("n_components", 5)
    covariance_type = hyperparameters.get("covariance_type", "full")
    max_iter = hyperparameters.get("max_iter", 100)
    random_state = hyperparameters.get("random_state", 42)
    
    # Prepare data
    X = data.select_dtypes(include=[np.number])
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state
    )
    
    labels = gmm.fit_predict(X_scaled)
    
    # Calculate metrics
    metrics = {}
    if len(set(labels)) > 1:  # Only calculate if we have more than one cluster
        metrics["silhouette_score"] = float(silhouette_score(X_scaled, labels))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
        
    # Include model-specific metrics
    metrics["aic"] = float(gmm.aic(X_scaled))
    metrics["bic"] = float(gmm.bic(X_scaled))
    
    # Count points per cluster
    cluster_counts = {}
    for label in np.unique(labels):
        cluster_counts[int(label)] = int(np.sum(labels == label))
    
    # Prepare output
    return {
        "labels": labels.tolist(),
        "means": gmm.means_.tolist(),
        "covariances": [cov.tolist() for cov in gmm.covariances_],
        "weights": gmm.weights_.tolist(),
        "cluster_counts": cluster_counts,
        "metrics": metrics
    }