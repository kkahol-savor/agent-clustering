"""
Clustering Agent: FastAPI application for running clustering algorithms.
"""

import os
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Try importing directly from files
try:
    from k_means import run_kmeans
    from db_scan import run_dbscan
    from hdbscan_algo import run_hdbscan
    from gmm import run_gmm
    from hierarchical import run_hierarchical
    from data_processor import DataProcessor
    from job_manager import JobManager
except ImportError:
    # If direct import fails, try from directories (with CORRECT filenames)
    try:
        from algorithms.k_means import run_kmeans
        from algorithms.db_scan import run_dbscan
        from algorithms.hdbscan_algo import run_hdbscan
        from algorithms.gmm import run_gmm
        from algorithms.hierarchical import run_hierarchical
        from core.data_processor import DataProcessor
        from core.job_manager import JobManager
    except ImportError:
        # If that fails too, something is wrong with the file structure
        logging.error("Failed to import required modules. Check file structure and naming.")
        raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(title="Clustering Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Data directory for temporary storage
DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize job manager and data processor
job_manager = JobManager()
data_processor = DataProcessor()

# Algorithm mapping
ALGORITHM_FUNCTIONS = {
    "kmeans": run_kmeans,
    "dbscan": run_dbscan,
    "hdbscan": run_hdbscan,
    "gmm": run_gmm,
    "agglomerative": run_hierarchical
}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Clustering Agent is running"}

@app.get("/api/v1/clustering/status/{job_id}")
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

@app.get("/api/v1/clustering/results/{job_id}")
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

@app.post("/api/v1/clustering/run")
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

if __name__ == "__main__":
    import uvicorn
    
    # Determine port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Clustering Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)