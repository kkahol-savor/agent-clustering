"""
K-Means clustering implementation for the clustering agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)

def run_kmeans(data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run K-Means clustering on the provided data.
    
    Args:
        data: Pandas DataFrame with features
        hyperparameters: Dictionary of hyperparameters for K-Means
        
    Returns:
        results: Dictionary with clustering results
    """
    logger.info(f"Starting K-Means clustering with hyperparameters: {hyperparameters}")
    
    # Extract hyperparameters with defaults
    n_clusters = hyperparameters.get("n_clusters", 5)
    init = hyperparameters.get("init", "k-means++")
    max_iter = hyperparameters.get("max_iter", 300)
    n_init = hyperparameters.get("n_init", 10)
    random_state = hyperparameters.get("random_state", 42)
    
    # Validate n_clusters
    try:
        n_clusters = int(n_clusters)
        if n_clusters < 2:
            logger.warning(f"Invalid n_clusters: {n_clusters}, using default value 2")
            n_clusters = 2
    except (ValueError, TypeError):
        logger.warning(f"Invalid n_clusters: {n_clusters}, using default value 5")
        n_clusters = 5
    
    # Prepare data
    X = data.select_dtypes(include=[np.number])
    
    # Log data shape
    logger.info(f"Numeric data shape: {X.shape}")
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    logger.info("Standardizing features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check for enough data
    if len(X) < n_clusters:
        logger.warning(f"Not enough data points ({len(X)}) for requested clusters ({n_clusters})")
        n_clusters = min(len(X) - 1, 2)
        logger.info(f"Adjusted n_clusters to {n_clusters}")
    
    # Run K-Means
    logger.info(f"Running K-Means with n_clusters={n_clusters}, init={init}, max_iter={max_iter}")
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state
    )
    
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    # Log convergence information
    logger.info(f"K-Means converged: {kmeans.n_iter_} iterations (max: {max_iter})")
    
    # Calculate metrics
    metrics = {}
    metrics["inertia"] = float(kmeans.inertia_)
    
    try:
        if len(set(labels)) > 1:  # Only calculate if we have more than one cluster
            metrics["silhouette_score"] = float(silhouette_score(X_scaled, labels))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
            logger.info(f"Metrics calculated: silhouette={metrics['silhouette_score']:.4f}, davies_bouldin={metrics['davies_bouldin_score']:.4f}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
    
    # Count points per cluster
    cluster_counts = {}
    for label in np.unique(labels):
        cluster_counts[int(label)] = int(np.sum(labels == label))
    
    logger.info(f"Cluster distribution: {cluster_counts}")
    
    # Prepare output
    return {
        "labels": labels.tolist(),
        "centroids": centroids.tolist(),
        "cluster_counts": cluster_counts,
        "metrics": metrics
    }