"""
Evaluator & Optimizer Agent: FastAPI application for evaluating clustering results.
"""

import os
import json
import logging
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from apscheduler.schedulers.background import BackgroundScheduler
import httpx
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(title="Evaluator & Optimizer API")

# Output directory for reports
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Agent 1 URL
AGENT1_URL = os.environ.get("AGENT1_URL", "http://clustering-agent:8000")

# Simple in-memory evaluations storage
evaluations = {}

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Run when the application shuts down."""
    scheduler.shutdown()
    logger.info("APScheduler shut down")

# Simple PDF generator
class SimpleReport(FPDF):
    """Basic PDF report generator."""
    
    def header(self):
        """Add header to each page."""
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Clustering Analysis Report", 0, 1, "C")
        self.ln(10)
    
    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
    
    def chapter_title(self, title):
        """Add a chapter title."""
        self.set_font("Arial", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, "L", 1)
        self.ln(4)
    
    def chapter_body(self, body):
        """Add chapter body text."""
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 5, body)
        self.ln()

async def generate_simple_report(eval_id: str, job_id: str, results: Dict[str, Any]) -> str:
    """Generate a simple PDF report for clustering results."""
    report_path = os.path.join(REPORTS_DIR, f"clustering_report_{eval_id}.pdf")
    
    # Create PDF
    pdf = SimpleReport()
    pdf.add_page()
    
    # Add content
    pdf.chapter_title("Clustering Results")
    pdf.chapter_body(f"Job ID: {job_id}\nEvaluation ID: {eval_id}")
    
    # Add metrics if available
    if "metrics" in results:
        pdf.chapter_title("Quality Metrics")
        metrics_text = ""
        for name, value in results["metrics"].items():
            metrics_text += f"{name}: {value}\n"
        pdf.chapter_body(metrics_text)
    
    # Save PDF
    pdf.output(report_path)
    
    return report_path

async def process_evaluation(eval_id: str, job_id: str):
    """Process a clustering evaluation."""
    try:
        # Update status
        evaluations[eval_id]["status"] = "in_progress"
        
        # Get results from Agent 1
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AGENT1_URL}/api/v1/clustering/results/{job_id}")
            
            if response.status_code != 200:
                evaluations[eval_id]["status"] = "error"
                evaluations[eval_id]["error"] = f"Error getting results: {response.text}"
                return
            
            results_data = response.json()
            
            if results_data.get("status") != "completed":
                evaluations[eval_id]["status"] = "error"
                evaluations[eval_id]["error"] = f"Job not completed: {results_data.get('status')}"
                return
            
            results = results_data.get("results", {})
            
            # Generate report
            evaluations[eval_id]["status"] = "generating_report"
            report_path = await generate_simple_report(eval_id, job_id, results)
            
            # Update evaluation
            evaluations[eval_id]["status"] = "completed"
            evaluations[eval_id]["report_path"] = report_path
    
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        evaluations[eval_id]["status"] = "error"
        evaluations[eval_id]["error"] = str(e)

# API Routes
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/v1/evaluator/evaluate")
async def evaluate_clustering(background_tasks: BackgroundTasks, job_id: str):
    """Start an evaluation for a clustering job."""
    try:
        # Create evaluation
        eval_id = str(uuid.uuid4())
        evaluations[eval_id] = {
            "job_id": job_id,
            "status": "created",
            "report_path": None,
            "error": None
        }
        
        # Run evaluation in background
        background_tasks.add_task(process_evaluation, eval_id, job_id)
        
        return {"eval_id": eval_id, "status": "evaluation_started"}
    
    except Exception as e:
        logger.error(f"Error starting evaluation: {str(e)}")
        return {"error": str(e)}

@app.get("/api/v1/evaluator/status/{eval_id}")
async def get_evaluation_status(eval_id: str):
    """Get the status of an evaluation."""
    if eval_id not in evaluations:
        raise HTTPException(404, f"Evaluation {eval_id} not found")
    
    return {
        "eval_id": eval_id,
        "status": evaluations[eval_id]["status"]
    }

@app.get("/api/v1/evaluator/report/{eval_id}")
async def get_report(eval_id: str):
    """Get the report for an evaluation."""
    if eval_id not in evaluations:
        raise HTTPException(404, f"Evaluation {eval_id} not found")
    
    eval_data = evaluations[eval_id]
    
    if eval_data["status"] != "completed":
        raise HTTPException(400, f"Report not ready: {eval_data['status']}")
    
    if not eval_data.get("report_path"):
        raise HTTPException(404, "Report not found")
    
    return FileResponse(
        path=eval_data["report_path"],
        filename=f"clustering_report_{eval_id}.pdf",
        media_type="application/pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)