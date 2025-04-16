"""
Evaluation functionality for assessing clustering results.
"""

import httpx
import numpy as np
import logging
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

async def evaluate_clustering(job_id: str, agent1_url: str) -> Dict[str, Any]:
    """
    Evaluate clustering results from Agent 1.
    
    Args:
        job_id: Clustering job ID
        agent1_url: URL for Agent 1 API
        
    Returns:
        evaluation_results: Dictionary with evaluation results
    """
    # Fetch clustering results from Agent 1
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{agent1_url}/api/v1/clustering/results/{job_id}")
        
    if response.status_code != 200:
        raise Exception(f"Failed to fetch clustering results: {response.text}")
        
    data = response.json()
    
    if data.get("status") != "completed" or not data.get("results"):
        raise Exception(f"Clustering job not complete or no results available")
    
    # Extract results
    clustering_results = data["results"]
    
    # Analyze the metrics
    metrics = clustering_results.get("metrics", {})
    
    # Create evaluation summary
    evaluation = {
        "algorithm": data.get("algorithm", "unknown"),
        "metrics_analysis": analyze_metrics(metrics),
        "clusters_analysis": analyze_clusters(clustering_results),
        "original_metrics": metrics
    }
    
    return evaluation

def analyze_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the clustering metrics to provide interpretation.
    
    Args:
        metrics: Dictionary of clustering metrics
        
    Returns:
        analysis: Dictionary with analysis of metrics
    """
    analysis = {}
    
    # Silhouette score interpretation
    if "silhouette_score" in metrics:
        score = metrics["silhouette_score"]
        if score > 0.7:
            quality = "excellent"
        elif score > 0.5:
            quality = "good"
        elif score > 0.3:
            quality = "fair"
        else:
            quality = "poor"
            
        analysis["silhouette_interpretation"] = {
            "value": score,
            "quality": quality,
            "explanation": (
                f"A silhouette score of {score:.4f} indicates {quality} cluster separation. "
                f"Values close to 1 indicate well-separated clusters."
            )
        }
    
    # Davies-Bouldin interpretation (lower is better)
    if "davies_bouldin_score" in metrics:
        score = metrics["davies_bouldin_score"]
        if score < 0.5:
            quality = "excellent"
        elif score < 0.7:
            quality = "good"
        elif score < 1.0:
            quality = "fair"
        else:
            quality = "poor"
            
        analysis["davies_bouldin_interpretation"] = {
            "value": score,
            "quality": quality,
            "explanation": (
                f"A Davies-Bouldin index of {score:.4f} indicates {quality} cluster separation. "
                f"Lower values indicate better clustering."
            )
        }
    
    # Calinski-Harabasz interpretation (higher is better)
    if "calinski_harabasz_score" in metrics:
        score = metrics["calinski_harabasz_score"]
        # The scale is relative, so use a log scale for interpretation
        log_score = np.log10(max(score, 1))
        
        if log_score > 3:  # > 1000
            quality = "excellent"
        elif log_score > 2:  # > 100
            quality = "good"
        elif log_score > 1:  # > 10
            quality = "fair"
        else:
            quality = "poor"
            
        analysis["calinski_harabasz_interpretation"] = {
            "value": score,
            "quality": quality,
            "explanation": (
                f"A Calinski-Harabasz score of {score:.2f} indicates {quality} cluster definition. "
                f"Higher values indicate better clustering."
            )
        }
    
    # Inertia interpretation (K-Means specific)
    if "inertia" in metrics:
        analysis["inertia_interpretation"] = {
            "value": metrics["inertia"],
            "explanation": (
                f"Inertia of {metrics['inertia']:.2f} represents the sum of squared distances "
                f"of samples to their closest cluster center. Lower values are better, "
                f"but this metric is not normalized and depends on the dataset size."
            )
        }
    
    # AIC/BIC interpretation (GMM specific)
    if "aic" in metrics and "bic" in metrics:
        analysis["information_criteria_interpretation"] = {
            "aic": metrics["aic"],
            "bic": metrics["bic"],
            "explanation": (
                f"AIC ({metrics['aic']:.2f}) and BIC ({metrics['bic']:.2f}) are information criteria "
                f"that balance model fit and complexity. Lower values indicate better models. "
                f"BIC penalizes model complexity more strongly than AIC."
            )
        }
    
    return analysis

def analyze_clusters(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the clustering results to provide insights about the clusters.
    
    Args:
        results: Dictionary with clustering results
        
    Returns:
        analysis: Dictionary with analysis of clusters
    """
    analysis = {}
    
    # Extract cluster information
    labels = results.get("labels", [])
    
    if not labels:
        return {"error": "No cluster labels found in results"}
    
    # Get unique labels and their counts
    if "cluster_counts" in results:
        cluster_counts = results["cluster_counts"]
    else:
        # Compute counts if not provided
        cluster_counts = {}
        for label in set(labels):
            cluster_counts[str(label)] = labels.count(label)
    
    # Calculate total points
    total_points = sum(cluster_counts.values())
    
    # Convert counts to percentages
    cluster_distribution = {}
    for label, count in cluster_counts.items():
        percentage = (count / total_points) * 100
        cluster_distribution[label] = {
            "count": count,
            "percentage": percentage
        }
    
    # Identify outliers/noise points if present
    if "-1" in cluster_distribution or "-1.0" in cluster_distribution:
        noise_key = "-1" if "-1" in cluster_distribution else "-1.0"
        noise_percentage = cluster_distribution[noise_key]["percentage"]
        noise_count = cluster_distribution[noise_key]["count"]
        
        if noise_percentage > 50:
            noise_assessment = "Very high"
            noise_impact = "The clustering has classified most points as noise, suggesting the algorithm may not be appropriate for this data or parameters need adjustment."
        elif noise_percentage > 30:
            noise_assessment = "High"
            noise_impact = "A substantial portion of data points are considered noise. Consider adjusting parameters or trying a different algorithm."
        elif noise_percentage > 10:
            noise_assessment = "Moderate"
            noise_impact = "Some data points are classified as noise, which is expected in density-based clustering of noisy datasets."
        else:
            noise_assessment = "Low"
            noise_impact = "Few data points are classified as noise, suggesting the algorithm is capturing most patterns well."
            
        analysis["noise_analysis"] = {
            "count": noise_count,
            "percentage": noise_percentage,
            "assessment": noise_assessment,
            "impact": noise_impact
        }
    
    # Analyze distribution balance
    if len(cluster_distribution) > 1:
        # Remove noise cluster from balance analysis
        balance_distribution = {k: v for k, v in cluster_distribution.items() if k != "-1" and k != "-1.0"}
        
        if balance_distribution:
            percentages = [v["percentage"] for v in balance_distribution.values()]
            max_percentage = max(percentages)
            min_percentage = min(percentages)
            ratio = max_percentage / max(min_percentage, 0.1)  # Avoid division by very small numbers
            
            if ratio > 10:
                balance_assessment = "Highly imbalanced"
                balance_impact = "Clusters have very uneven sizes, which may indicate dominant patterns or suboptimal parameter selection."
            elif ratio > 5:
                balance_assessment = "Imbalanced"
                balance_impact = "Clusters show significant size differences. This may be natural for the dataset or suggest parameter tuning may help."
            elif ratio > 2:
                balance_assessment = "Moderately balanced"
                balance_impact = "Clusters have some size variation, which is common in many real-world datasets."
            else:
                balance_assessment = "Well balanced"
                balance_impact = "Clusters are relatively similar in size, suggesting evenly distributed patterns in the data."
                
            analysis["balance_analysis"] = {
                "max_to_min_ratio": ratio,
                "assessment": balance_assessment,
                "impact": balance_impact
            }
    
    # Add basic statistics
    analysis["num_clusters"] = len(cluster_distribution) - (1 if "-1" in cluster_distribution or "-1.0" in cluster_distribution else 0)
    analysis["total_points"] = total_points
    analysis["cluster_distribution"] = cluster_distribution
    
    return analysis