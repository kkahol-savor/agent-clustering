"""
API routes for the evaluator agent, implementing the specification
from the technical documentation.
"""

import os
import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.evaluation import evaluate_clustering
from core.optimization import optimize_hyperparameters
from core.pdf_generator import generate_report
from core.notification import send_notification
from core.scheduler import schedule_evaluation
from api.models import EvaluationRequest, EvaluationResponse, StatusResponse

# Initialize router
router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

# Track evaluations (in-memory for simplicity)
evaluations = {}

# Directory for reports
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Agent 1 URL
AGENT1_URL = os.environ.get("AGENT1_URL", "http://clustering-agent:8000")

@router.post("/evaluate", response_model=EvaluationResponse)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Start a clustering evaluation with optional optimization and report generation.
    
    Args:
        request: Evaluation request containing job_id, notification settings, and optimization config
        background_tasks: FastAPI background tasks for asynchronous processing
    
    Returns:
        Evaluation response with eval_id and status
    """
    try:
        # Create evaluation ID and record
        eval_id = str(uuid.uuid4())
        
        evaluations[eval_id] = {
            "status": "evaluation_started",
            "job_id": request.job_id,
            "notify_sms": request.notify_sms.dict() if request.notify_sms else None,
            "optimize": request.optimize,
            "optimization_config": request.optimization_config.dict() if request.optimization_config else None,
            "results": None,
            "report_path": None,
            "best_params": None
        }
        
        # Run evaluation in the background
        background_tasks.add_task(
            process_evaluation,
            eval_id,
            request.job_id,
            request.optimize,
            request.optimization_config.dict() if request.optimization_config else None,
            request.notify_sms.dict() if request.notify_sms else None
        )
        
        return {"eval_id": eval_id, "status": "evaluation_started"}
    
    except Exception as e:
        logger.error(f"Error starting evaluation: {str(e)}")
        raise HTTPException(500, f"Error starting evaluation: {str(e)}")

@router.get("/status/{eval_id}", response_model=StatusResponse)
async def get_evaluation_status(eval_id: str):
    """
    Check the status of an evaluation.
    
    Args:
        eval_id: Evaluation ID to check
    
    Returns:
        Status response with current status and best parameters if available
    """
    try:
        if eval_id not in evaluations:
            raise HTTPException(404, f"Evaluation with ID {eval_id} not found")
            
        response = {
            "eval_id": eval_id,
            "status": evaluations[eval_id]["status"]
        }
        
        # Include best parameters if available
        if evaluations[eval_id].get("best_params"):
            response["best_params"] = evaluations[eval_id]["best_params"]
            
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation status: {str(e)}")
        raise HTTPException(500, f"Error getting evaluation status: {str(e)}")

@router.get("/report/{eval_id}")
async def get_report(eval_id: str):
    """
    Get the PDF report for an evaluation.
    
    Args:
        eval_id: Evaluation ID to get the report for
    
    Returns:
        PDF file as a download
    """
    try:
        if eval_id not in evaluations:
            raise HTTPException(404, f"Evaluation with ID {eval_id} not found")
            
        eval_data = evaluations[eval_id]
        
        if eval_data["status"] != "completed":
            raise HTTPException(400, f"Report not ready. Current status: {eval_data['status']}")
            
        if not eval_data.get("report_path"):
            raise HTTPException(404, "Report not found")
            
        return FileResponse(
            path=eval_data["report_path"],
            filename=f"clustering_report_{eval_id}.pdf",
            media_type="application/pdf"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report: {str(e)}")
        raise HTTPException(500, f"Error retrieving report: {str(e)}")

async def process_evaluation(
    eval_id: str,
    job_id: str,
    optimize: bool,
    optimization_config: Optional[Dict[str, Any]],
    notify_settings: Optional[Dict[str, Any]]
):
    """
    Process an evaluation request asynchronously.
    
    Args:
        eval_id: Evaluation ID
        job_id: Clustering job ID to evaluate
        optimize: Whether to optimize hyperparameters
        optimization_config: Configuration for hyperparameter optimization
        notify_settings: Settings for SMS notification
    """
    try:
        # Update status to in_progress
        evaluations[eval_id]["status"] = "in_progress"
        
        # Step 1: Evaluate the clustering results
        evaluation_results = await evaluate_clustering(job_id, AGENT1_URL)
        evaluations[eval_id]["results"] = evaluation_results
        
        # Step 2: Optimize hyperparameters if requested
        if optimize and optimization_config:
            evaluations[eval_id]["status"] = "optimizing"
            
            best_params, optimized_results = await optimize_hyperparameters(
                job_id, 
                evaluation_results["algorithm"],
                optimization_config,
                AGENT1_URL
            )
            
            evaluations[eval_id]["best_params"] = best_params
            evaluations[eval_id]["results"]["optimized"] = optimized_results
        
        # Step 3: Generate PDF report
        evaluations[eval_id]["status"] = "generating_report"
        
        report_path = await generate_report(
            eval_id,
            job_id,
            evaluations[eval_id]["results"],
            optimize,
            evaluations[eval_id].get("best_params"),
            REPORTS_DIR
        )
        
        evaluations[eval_id]["report_path"] = report_path
        
        # Step 4: Send notification if requested
        if notify_settings and notify_settings.get("enabled"):
            await send_notification(
                eval_id,
                notify_settings.get("phone_number"),
                notify_settings.get("carrier_gateway")
            )
        
        # Update status to completed
        evaluations[eval_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Error in process_evaluation: {str(e)}")
        evaluations[eval_id]["status"] = "error"
        evaluations[eval_id]["error"] = str(e)