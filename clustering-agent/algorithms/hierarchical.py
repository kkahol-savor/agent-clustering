"""
Hierarchical (Agglomerative) clustering implementation for the clustering agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

def run_hierarchical(data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Hierarchical (Agglomerative) clustering on the provided data.
    
    Args:
        data: Pandas DataFrame with features
        hyperparameters: Dictionary of hyperparameters for Agglomerative clustering
        
    Returns:
        results: Dictionary with clustering results
    """
    # Extract hyperparameters with defaults
    n_clusters = hyperparameters.get("n_clusters", 5)
    linkage = hyperparameters.get("linkage", "ward")
    affinity = hyperparameters.get("affinity", "euclidean")
    
    # Check for incompatible combinations
    if linkage == "ward" and affinity != "euclidean":
        affinity = "euclidean"  # Ward linkage only works with euclidean distance
    
    # Prepare data
    X = data.select_dtypes(include=[np.number])
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run Agglomerative Clustering
    agglo = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        affinity=affinity
    )
    
    labels = agglo.fit_predict(X_scaled)
    
    # Calculate metrics
    metrics = {}
    if len(set(labels)) > 1:  # Only calculate if we have more than one cluster
        metrics["silhouette_score"] = float(silhouette_score(X_scaled, labels))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
    
    # Count points per cluster
    cluster_counts = {}
    for label in np.unique(labels):
        cluster_counts[int(label)] = int(np.sum(labels == label))
    
    # Prepare output
    return {
        "labels": labels.tolist(),
        "cluster_counts": cluster_counts,
        "metrics": metrics
    }