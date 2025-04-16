"""
Hyperparameter optimization functionality using Optuna.
"""

import httpx
import json
import logging
import optuna
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Algorithm parameter ranges for optimization
ALGORITHM_PARAM_RANGES = {
    "kmeans": {
        "n_clusters": (2, 20),
        "init": ["k-means++", "random"],
        "max_iter": (100, 500)
    },
    "dbscan": {
        "eps": (0.1, 2.0),
        "min_samples": (2, 20)
    },
    "hdbscan": {
        "min_cluster_size": (2, 20),
        "min_samples": (1, 20),
        "cluster_selection_epsilon": (0.0, 1.0)
    },
    "gmm": {
        "n_components": (2, 20),
        "covariance_type": ["full", "tied", "diag", "spherical"]
    },
    "agglomerative": {
        "n_clusters": (2, 20),
        "linkage": ["ward", "complete", "average", "single"],
        "affinity": ["euclidean", "manhattan", "cosine"]
    }
}

async def optimize_hyperparameters(
    job_id: str, 
    algorithm: str, 
    optimization_config: Dict[str, Any],
    agent1_url: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Optimize hyperparameters for a clustering algorithm.
    
    Args:
        job_id: Original clustering job ID
        algorithm: Algorithm name
        optimization_config: Configuration for optimization
        agent1_url: URL for Agent 1 API
        
    Returns:
        Tuple of (best_params, best_results)
    """
    max_iterations = optimization_config.get("max_iterations", 10)
    search_algorithm = optimization_config.get("search_algorithm", "optuna")
    
    # Get parameter ranges for the selected algorithm
    param_ranges = ALGORITHM_PARAM_RANGES.get(algorithm, {})
    
    if not param_ranges:
        raise ValueError(f"Unsupported algorithm for optimization: {algorithm}")
    
    # Retrieve original data info
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{agent1_url}/api/v1/clustering/results/{job_id}")
        
    if response.status_code != 200:
        raise Exception(f"Failed to fetch original clustering results: {response.text}")
        
    original_data = response.json()
    
    # Define the objective function for Optuna
    def objective(trial):
        # Generate hyperparameters based on the algorithm
        hyperparameters = {}
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Numeric parameter
                start, end = param_range
                if isinstance(start, int) and isinstance(end, int):
                    hyperparameters[param_name] = trial.suggest_int(param_name, start, end)
                else:
                    hyperparameters[param_name] = trial.suggest_float(param_name, start, end)
            elif isinstance(param_range, list):
                # Categorical parameter
                hyperparameters[param_name] = trial.suggest_categorical(param_name, param_range)
        
        # Special cases and constraints
        if algorithm == "agglomerative" and hyperparameters.get("linkage") == "ward":
            hyperparameters["affinity"] = "euclidean"  # Ward only works with euclidean
        
        # Run clustering with the trial hyperparameters
        try:
            response = httpx.post(
                f"{agent1_url}/api/v1/clustering/run",
                json={
                    "data_url": original_data.get("data_url", ""),  # This might not be available
                    "algorithm": algorithm,
                    "hyperparameters": hyperparameters
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Error in clustering run: {response.text}")
                return float("-inf")  # Return worst possible score
                
            trial_job_id = response.json().get("job_id")
            
            # Wait for the job to complete
            while True:
                status_response = httpx.get(f"{agent1_url}/api/v1/clustering/status/{trial_job_id}")
                status_data = status_response.json()
                
                if status_data.get("status") == "completed":
                    break
                elif status_data.get("status") == "error":
                    logger.error(f"Error in trial job: {status_data}")
                    return float("-inf")
                
                import time
                time.sleep(1)  # Wait before checking again
                
            # Get the results
            results_response = httpx.get(f"{agent1_url}/api/v1/clustering/results/{trial_job_id}")
            results_data = results_response.json()
            
            if results_data.get("status") != "completed" or not results_data.get("results"):
                logger.error(f"No results available for trial job")
                return float("-inf")
                
            # Extract metrics
            metrics = results_data.get("results", {}).get("metrics", {})
            
            # Determine the best metric to optimize
            score = None
            
            if "silhouette_score" in metrics:
                # Higher is better
                score = metrics["silhouette_score"]
            elif "davies_bouldin_score" in metrics:
                # Lower is better, so negate
                score = -metrics["davies_bouldin_score"]
            elif "calinski_harabasz_score" in metrics:
                # Higher is better
                score = metrics["calinski_harabasz_score"]
            elif "inertia" in metrics:
                # Lower is better, so negate
                score = -metrics["inertia"]
            
            if score is None:
                logger.error(f"No suitable metrics found for optimization")
                return float("-inf")
                
            return score
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return float("-inf")
    
    # Create and run the study
    if search_algorithm == "optuna":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=max_iterations)
        
        best_params = study.best_params
        
        # Run one final time with the best parameters to get the results
        final_response = httpx.post(
            f"{agent1_url}/api/v1/clustering/run",
            json={
                "data_url": original_data.get("data_url", ""),
                "algorithm": algorithm,
                "hyperparameters": best_params
            },
            timeout=60
        )
        
        final_job_id = final_response.json().get("job_id")
        
        # Wait for completion
        while True:
            status_response = httpx.get(f"{agent1_url}/api/v1/clustering/status/{final_job_id}")
            if status_response.json().get("status") == "completed":
                break
            import time
            time.sleep(1)
            
        # Get final results
        final_results_response = httpx.get(f"{agent1_url}/api/v1/clustering/results/{final_job_id}")
        final_results = final_results_response.json().get("results", {})
        
        return best_params, final_results
    else:
        raise ValueError(f"Unsupported search algorithm: {search_algorithm}")