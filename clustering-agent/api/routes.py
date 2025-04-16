"""
API routes for the clustering agent, implementing the specification
from the technical documentation.
"""

import os
import uuid
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel

from core.job_manager import JobManager
from core.data_processor import DataProcessor

# Fix imports to match the actual file structure
# Use relative imports from the current package
try:
    # Try the original import paths first
    from algorithms.kmeans import run_kmeans
    from algorithms.dbscan import run_dbscan
    from algorithms.hdbscan_algo import run_hdbscan
    from algorithms.gmm import run_gmm
    from algorithms.hierarchical import run_hierarchical
except ModuleNotFoundError:
    # If that fails, try alternative paths
    try:
        # Try from parent directory
        from ..algorithms.k_means import run_kmeans
        from ..algorithms.db_scan import run_dbscan
        from ..algorithms.hdbscan_algo import run_hdbscan
        from ..algorithms.gmm import run_gmm
        from ..algorithms.hierarchical import run_hierarchical
    except ModuleNotFoundError:
        # Last resort: import directly by filename without package structure
        import sys
        import os.path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from k_means import run_kmeans
        from db_scan import run_dbscan
        from hdbscan_algo import run_hdbscan
        from gmm import run_gmm
        from hierarchical import run_hierarchical

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize job manager and data processor
job_manager = JobManager()
data_processor = DataProcessor()

# Define data models
class Hyperparameters(BaseModel):
    """Model for hyperparameters as key-value pairs."""
    params: Dict[str, Any]

class JobRequest(BaseModel):
    """Request model for starting a clustering job."""
    data_url: Optional[str] = None
    algorithm: str
    hyperparameters: Dict[str, Any]

class JobResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str

class ResultsResponse(BaseModel):
    """Response model for clustering results."""
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None

# Algorithm mapping
ALGORITHM_FUNCTIONS = {
    "kmeans": run_kmeans,
    "dbscan": run_dbscan,
    "hdbscan": run_hdbscan,
    "gmm": run_gmm,
    "agglomerative": run_hierarchical
}

@router.post("/run")
async def run_clustering(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    hyperparameters: str = Form(...)
):
    """Start a clustering job with the uploaded file."""
    try:
        # Log received file information
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Raw hyperparameters: {hyperparameters}")
        
        # Handle empty hyperparameters
        if not hyperparameters.strip():
            hyperparameters = "{}"
            
        try:
            params = json.loads(hyperparameters)
            logger.info(f"Parsed hyperparameters: {params}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON format: {str(e)}"
            )
            
        # Save the uploaded file
        file_path = await data_processor.save_uploaded_file(file)
        logger.info(f"File saved to: {file_path}")
        
        # Create a new job
        job_id = job_manager.create_job(algorithm, params, file_path)
        logger.info(f"Created job: {job_id}")
        
        # Execute clustering in the background
        background_tasks.add_task(execute_clustering, job_id, algorithm, params, file_path)
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error in run_clustering endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a clustering job."""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job with ID {job_id} not found")
            
        return {"job_id": job_id, "status": job["status"]}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(500, f"Error getting job status: {str(e)}")

@router.get("/results/{job_id}", response_model=ResultsResponse)
async def get_job_results(job_id: str):
    """Get the results of a completed clustering job."""
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(404, f"Job with ID {job_id} not found")
            
        if job["status"] != "completed":
            return {"job_id": job_id, "status": job["status"]}
            
        return {
            "job_id": job_id,
            "status": "completed",
            "results": job["results"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(500, f"Error getting job results: {str(e)}")

async def execute_clustering(
    job_id: str, 
    algorithm: str, 
    hyperparameters: Dict[str, Any], 
    file_path: str
):
    """Execute the clustering algorithm and update the job status."""
    try:
        # Update job status to in_progress
        job_manager.update_job_status(job_id, "in_progress")
        logger.info(f"Job {job_id} status updated to in_progress")
        
        # Load data
        try:
            data = data_processor.load_data(file_path)
            logger.info(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")
            # Log column names and types
            for col in data.columns:
                logger.info(f"Column: {col}, Type: {data[col].dtype}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            job_manager.update_job_status(job_id, "error")
            job_manager.update_job_error(job_id, f"Error loading data: {str(e)}")
            return
        
        # Check if the algorithm is supported
        if algorithm not in ALGORITHM_FUNCTIONS:
            logger.error(f"Unsupported algorithm: {algorithm}")
            job_manager.update_job_status(job_id, "error")
            job_manager.update_job_error(job_id, f"Unsupported algorithm: {algorithm}")
            return
            
        # Run the appropriate clustering algorithm
        try:
            algorithm_func = ALGORITHM_FUNCTIONS[algorithm]
            results = algorithm_func(data, hyperparameters)
            logger.info(f"Clustering completed with results: {results.keys()}")
        except Exception as e:
            logger.error(f"Error in clustering algorithm: {str(e)}")
            job_manager.update_job_status(job_id, "error")
            job_manager.update_job_error(job_id, f"Error in clustering algorithm: {str(e)}")
            return
        
        # Update job with results
        job_manager.update_job_results(job_id, results)
        job_manager.update_job_status(job_id, "completed")
        logger.info(f"Job {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Error in execute_clustering: {str(e)}")
        job_manager.update_job_status(job_id, "error")
        job_manager.update_job_error(job_id, str(e))