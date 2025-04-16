"""
Job scheduling functionality using APScheduler.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import httpx

# Configure logging
logger = logging.getLogger(__name__)

# Agent 1 URL
AGENT1_URL = os.environ.get("AGENT1_URL", "http://clustering-agent:8000")

def initialize_scheduler() -> BackgroundScheduler:
    """
    Initialize the APScheduler for scheduled evaluations.
    
    Returns:
        scheduler: Configured APScheduler instance
    """
    scheduler = BackgroundScheduler()
    return scheduler

async def run_evaluation_pipeline(
    data_url: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    optimize: bool = False,
    notify_sms: Optional[Dict[str, Any]] = None
):
    """
    Run a complete clustering and evaluation pipeline.
    
    Args:
        data_url: URL to the data file
        algorithm: Clustering algorithm to use
        hyperparameters: Initial hyperparameters
        optimize: Whether to optimize hyperparameters
        notify_sms: SMS notification settings
    """
    try:
        logger.info(f"Starting scheduled evaluation pipeline for {algorithm} on {data_url}")
        
        # Step 1: Run clustering via Agent 1
        async with httpx.AsyncClient() as client:
            clustering_response = await client.post(
                f"{AGENT1_URL}/api/v1/clustering/run",
                json={
                    "data_url": data_url,
                    "algorithm": algorithm,
                    "hyperparameters": hyperparameters
                }
            )
            
            clustering_data = clustering_response.json()
            job_id = clustering_data.get("job_id")
            
            if not job_id:
                logger.error(f"Failed to start clustering job: {clustering_data}")
                return
                
            logger.info(f"Clustering job started with ID: {job_id}")
            
            # Wait for clustering to complete
            while True:
                status_response = await client.get(f"{AGENT1_URL}/api/v1/clustering/status/{job_id}")
                status_data = status_response.json()
                
                if status_data.get("status") == "completed":
                    break
                elif status_data.get("status") == "error":
                    logger.error(f"Error in clustering job: {status_data}")
                    return
                    
                import asyncio
                await asyncio.sleep(2)  # Wait before checking again
            
            # Step 2: Run evaluation via Agent 2 (assuming this is running on Agent 2)
            from api.models import EvaluationRequest, NotifySMS, OptimizationConfig
            
            notify_sms_obj = None
            if notify_sms:
                notify_sms_obj = NotifySMS(**notify_sms)
                
            optimization_config = None
            if optimize:
                optimization_config = OptimizationConfig(max_iterations=10, search_algorithm="optuna")
                
            eval_request = EvaluationRequest(
                job_id=job_id,
                notify_sms=notify_sms_obj,
                optimize=optimize,
                optimization_config=optimization_config
            )
            
            # Import the evaluation function
            from core.evaluation import evaluate_clustering
            from core.optimization import optimize_hyperparameters
            from core.pdf_generator import generate_report
            from core.notification import send_notification
            
            # We're already in Agent 2, so we can call the functions directly
            evaluation_results = await evaluate_clustering(job_id, AGENT1_URL)
            
            best_params = None
            if optimize:
                best_params, optimized_results = await optimize_hyperparameters(
                    job_id,
                    algorithm,
                    optimization_config.dict() if optimization_config else {},
                    AGENT1_URL
                )
                evaluation_results["optimized"] = optimized_results
                
            # Generate report
            eval_id = str(uuid.uuid4())
            report_path = await generate_report(
                eval_id,
                job_id,
                evaluation_results,
                optimize,
                best_params,
                os.environ.get("REPORTS_DIR", "reports")
            )
            
            # Send notification if requested
            if notify_sms_obj and notify_sms_obj.enabled:
                await send_notification(
                    eval_id,
                    notify_sms_obj.phone_number,
                    notify_sms_obj.carrier_gateway
                )
                
            logger.info(f"Scheduled evaluation pipeline completed successfully. Report: {report_path}")
            
    except Exception as e:
        logger.error(f"Error in scheduled evaluation pipeline: {str(e)}")

def schedule_evaluation(
    schedule_type: str,
    schedule_params: Dict[str, Any],
    data_url: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    optimize: bool = False,
    notify_sms: Optional[Dict[str, Any]] = None
) -> str:
    """
    Schedule a recurring evaluation pipeline.
    
    Args:
        schedule_type: Type of schedule ('interval' or 'cron')
        schedule_params: Parameters for the schedule
        data_url: URL to the data file
        algorithm: Clustering algorithm to use
        hyperparameters: Initial hyperparameters
        optimize: Whether to optimize hyperparameters
        notify_sms: SMS notification settings
        
    Returns:
        job_id: ID of the scheduled job
    """
    scheduler = initialize_scheduler()
    
    # Create a job ID
    job_id = str(uuid.uuid4())
    
    # Create the trigger based on schedule type
    if schedule_type == "interval":
        # Extract interval parameters with defaults
        days = schedule_params.get("days", 0)
        hours = schedule_params.get("hours", 0)
        minutes = schedule_params.get("minutes", 0)
        seconds = schedule_params.get("seconds", 0)
        
        trigger = IntervalTrigger(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
    elif schedule_type == "cron":
        # Extract cron parameters
        cron_expr = schedule_params.get("expression", "0 0 * * *")  # Default: daily at midnight
        
        trigger = CronTrigger.from_crontab(cron_expr)
    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}")
    
    # Add the job to the scheduler
    scheduler.add_job(
        func=run_evaluation_pipeline,
        trigger=trigger,
        args=[data_url, algorithm, hyperparameters, optimize, notify_sms],
        id=job_id,
        name=f"Clustering evaluation for {algorithm} on {data_url}",
        replace_existing=True
    )
    
    logger.info(f"Scheduled evaluation pipeline with job ID: {job_id}")
    return job_id