"""
Job management functionality for the clustering agent,
storing job metadata, status, and results.
"""

import os
import json
import uuid
from typing import Dict, Any, Optional
import threading
import logging

# Configure logging
logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages clustering jobs, their status, and results.
    For simplicity, this uses in-memory storage, but a production
    implementation would use a database.
    """
    
    def __init__(self):
        """Initialize the job manager with an empty jobs dictionary."""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
    def create_job(
        self, 
        algorithm: str, 
        hyperparameters: Dict[str, Any], 
        data_path: str
    ) -> str:
        """
        Create a new clustering job.
        
        Args:
            algorithm: Name of the clustering algorithm to use
            hyperparameters: Dictionary of hyperparameters for the algorithm
            data_path: Path to the data file
            
        Returns:
            job_id: Unique identifier for the job
        """
        job_id = str(uuid.uuid4())
        
        with self.lock:
            self.jobs[job_id] = {
                "algorithm": algorithm,
                "hyperparameters": hyperparameters,
                "data_path": data_path,
                "status": "created",
                "results": None,
                "error": None
            }
            
        return job_id
        
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the job details by ID.
        
        Args:
            job_id: The job identifier
            
        Returns:
            job: The job details dictionary, or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)
            
    def update_job_status(self, job_id: str, status: str) -> bool:
        """
        Update the status of a job.
        
        Args:
            job_id: The job identifier
            status: New status (created, in_progress, completed, error)
            
        Returns:
            success: True if the job was updated, False otherwise
        """
        with self.lock:
            if job_id not in self.jobs:
                return False
                
            self.jobs[job_id]["status"] = status
            return True
            
    def update_job_results(self, job_id: str, results: Dict[str, Any]) -> bool:
        """
        Update the results of a job.
        
        Args:
            job_id: The job identifier
            results: Dictionary of clustering results
            
        Returns:
            success: True if the job was updated, False otherwise
        """
        with self.lock:
            if job_id not in self.jobs:
                return False
                
            self.jobs[job_id]["results"] = results
            return True
            
    def update_job_error(self, job_id: str, error: str) -> bool:
        """
        Update the error message of a job.
        
        Args:
            job_id: The job identifier
            error: Error message
            
        Returns:
            success: True if the job was updated, False otherwise
        """
        with self.lock:
            if job_id not in self.jobs:
                return False
                
            self.jobs[job_id]["error"] = error
            return True