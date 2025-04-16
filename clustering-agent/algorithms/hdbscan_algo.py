## clustering-agent/algorithms/hdbscan_algo.py

"""
HDBSCAN clustering implementation for the clustering agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

def run_hdbscan(data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run HDBSCAN clustering on the provided data.
    
    Args:
        data: Pandas DataFrame with features
        hyperparameters: Dictionary of hyperparameters for HDBSCAN
        
    Returns:
        results: Dictionary with clustering results
    """
    # Extract hyperparameters with defaults
    min_cluster_size = hyperparameters.get("min_cluster_size", 5)
    min_samples = hyperparameters.get("min_samples", 5)
    cluster_selection_epsilon = hyperparameters.get("cluster_selection_epsilon", 0.0)
    
    # Prepare data
    X = data.select_dtypes(include=[np.number])
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    
    labels = hdbscan_clusterer.fit_predict(X_scaled)
    
    # Calculate metrics if we have more than one cluster (excluding noise)
    metrics = {}
    unique_labels = set(labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        # Only calculate silhouette if we have valid clusters (excluding noise points)
        non_noise_indices = labels != -1
        if np.sum(non_noise_indices) > 1 and len(set(labels[non_noise_indices])) > 1:
            metrics["silhouette_score"] = float(silhouette_score(
                X_scaled[non_noise_indices], 
                labels[non_noise_indices]
            ))
            
        # Other metrics require at least 2 clusters
        if len(set(labels)) > 1:
            try:
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
            except Exception:
                # Some metrics might fail with certain cluster arrangements
                pass
    
    # Include probabilities and outlier scores if available
    probabilities = None
    outlier_scores = None
    
    if hasattr(hdbscan_clusterer, 'probabilities_'):
        probabilities = hdbscan_clusterer.probabilities_.tolist()
        
    if hasattr(hdbscan_clusterer, 'outlier_scores_'):
        outlier_scores = hdbscan_clusterer.outlier_scores_.tolist()
    
    # Count points per cluster (including noise as -1)
    cluster_counts = {}
    for label in np.unique(labels):
        cluster_counts[int(label)] = int(np.sum(labels == label))
    
    # Prepare output
    result = {
        "labels": labels.tolist(),
        "cluster_counts": cluster_counts,
        "metrics": metrics
    }
    
    if probabilities:
        result["probabilities"] = probabilities
        
    if outlier_scores:
        result["outlier_scores"] = outlier_scores
        
    return result