"""
DBSCAN clustering implementation for the clustering agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)

def run_dbscan(data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run DBSCAN clustering on the provided data.
    
    Args:
        data: Pandas DataFrame with features
        hyperparameters: Dictionary of hyperparameters for DBSCAN
        
    Returns:
        results: Dictionary with clustering results
    """
    logger.info(f"Starting DBSCAN clustering with hyperparameters: {hyperparameters}")
    
    # Extract hyperparameters with defaults
    try:
        eps = float(hyperparameters.get("eps", 0.5))
        if eps <= 0:
            logger.warning(f"Invalid eps value: {eps}, using default 0.5")
            eps = 0.5
    except (ValueError, TypeError):
        logger.warning(f"Invalid eps value: {hyperparameters.get('eps')}, using default 0.5")
        eps = 0.5
        
    try:
        min_samples = int(hyperparameters.get("min_samples", 5))
        if min_samples < 1:
            logger.warning(f"Invalid min_samples value: {min_samples}, using default 5")
            min_samples = 5
    except (ValueError, TypeError):
        logger.warning(f"Invalid min_samples value: {hyperparameters.get('min_samples')}, using default 5")
        min_samples = 5
    
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
    
    # Run DBSCAN
    logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    
    labels = dbscan.fit_predict(X_scaled)
    
    # Check results
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Calculate metrics if we have more than one cluster (excluding noise)
    metrics = {}
    if n_clusters > 0:
        try:
            # Only calculate silhouette if we have valid clusters (excluding noise points)
            non_noise_mask = labels != -1
            non_noise_count = np.sum(non_noise_mask)
            
            if non_noise_count > 1 and len(set(labels[non_noise_mask])) > 1:
                metrics["silhouette_score"] = float(silhouette_score(
                    X_scaled[non_noise_mask], 
                    labels[non_noise_mask]
                ))
                logger.info(f"Silhouette score: {metrics['silhouette_score']:.4f}")
                
            # Other metrics require at least 2 clusters
            if len(set(labels)) > 1:
                try:
                    metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
                    metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
                    logger.info(f"Davies-Bouldin score: {metrics['davies_bouldin_score']:.4f}")
                    logger.info(f"Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.4f}")
                except Exception as e:
                    logger.warning(f"Error calculating some metrics: {str(e)}")
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
    else:
        logger.warning("No clusters found or only noise points, metrics cannot be calculated")
    
    # Count points per cluster (including noise as -1)
    cluster_counts = {}
    for label in unique_labels:
        cluster_counts[int(label)] = int(np.sum(labels == label))
    
    logger.info(f"Cluster distribution: {cluster_counts}")
    
    # Add noise percentage metric
    if len(labels) > 0:
        noise_percentage = (n_noise / len(labels)) * 100
        metrics["noise_percentage"] = float(noise_percentage)
        logger.info(f"Noise percentage: {noise_percentage:.2f}%")
    
    # Prepare output
    return {
        "labels": labels.tolist(),
        "cluster_counts": cluster_counts,
        "metrics": metrics
    }