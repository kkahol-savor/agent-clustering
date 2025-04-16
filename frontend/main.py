from fastapi import FastAPI, Request, Query, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
import os
import json
import httpx
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from query_openai import QueryOpenAi

# Configuration
AGENT1_URL = os.getenv("AGENT1_URL", "http://clustering-agent:8000")
AGENT2_URL = os.getenv("AGENT2_URL", "http://evaluator-agent:8001")
UPLOAD_DIR = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Mount static directory for CSS, JS, and images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def login(request: Request):
    '''Default application endpoint that renders the index.html template'''
    return templates.TemplateResponse("index.html", {"request": request, "username": "Guest"})

@app.get("/home", response_class=HTMLResponse)
async def auth_redirect(request: Request):
    '''Home endpoint that renders the index.html template'''
    return templates.TemplateResponse("index.html", {"request": request, "username": "Guest"})

# View report page
@app.get("/report/{eval_id}", response_class=HTMLResponse)
async def view_report(request: Request, eval_id: str):
    '''Render the report viewing template'''
    return templates.TemplateResponse("report.html", {"request": request, "report_id": eval_id})

# File upload endpoint
@app.post("/api/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    '''Handle file upload and save to the uploads directory'''
    if file.content_type not in ["text/csv", "application/vnd.ms-excel", 
                                "application/json", "text/plain"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Generate a unique filename to prevent collisions
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Store original filename as well
    with open(f"{file_path}.meta", "w") as f:
        f.write(file.filename)
    
    return {"filename": unique_filename, "original_filename": file.filename}

# Proxy to Agent 1 (Clustering Agent) endpoints
@app.post("/api/v1/clustering/run")
async def run_clustering(request: Request):
    '''Proxy the clustering request to Agent 1'''
    try:
        # Read the request body
        data = await request.json()
        
        # If data contains a filename, prepare the file for Agent 1
        if "data_filename" in data:
            file_path = os.path.join(UPLOAD_DIR, data["data_filename"])
            
            # Create a multipart form with the file
            files = {"file": open(file_path, "rb")}
            params = {
                "algorithm": data["algorithm"],
                "hyperparameters": json.dumps(data["hyperparameters"])
            }
            
            # Send request to Agent 1
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{AGENT1_URL}/api/v1/clustering/run",
                    files=files,
                    data=params
                )
            
            # Close the file
            files["file"].close()
            
            # Return the response from Agent 1
            return response.json()
        else:
            # Forward the request directly
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{AGENT1_URL}/api/v1/clustering/run",
                    json=data
                )
            return response.json()
    except Exception as e:
        logger.error(f"Error in run_clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/v1/clustering/status/{job_id}")
async def get_clustering_status(job_id: str):
    '''Proxy the status check request to Agent 1'''
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AGENT1_URL}/api/v1/clustering/status/{job_id}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in get_clustering_status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@app.get("/api/v1/clustering/results/{job_id}")
async def get_clustering_results(job_id: str):
    '''Proxy the results request to Agent 1'''
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AGENT1_URL}/api/v1/clustering/results/{job_id}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in get_clustering_results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

# Proxy to Agent 2 (Evaluator Agent) endpoints
@app.post("/api/v1/evaluator/evaluate")
async def evaluate_clustering(request: Request):
    '''Proxy the evaluation request to Agent 2'''
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{AGENT2_URL}/api/v1/evaluator/evaluate", json=data)
        return response.json()
    except Exception as e:
        logger.error(f"Error in evaluate_clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating clustering: {str(e)}")

@app.get("/api/v1/evaluator/status/{eval_id}")
async def get_evaluation_status(eval_id: str):
    '''Proxy the evaluation status check to Agent 2'''
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AGENT2_URL}/api/v1/evaluator/status/{eval_id}")
        return response.json()
    except Exception as e:
        logger.error(f"Error in get_evaluation_status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking evaluation status: {str(e)}")

@app.get("/api/v1/evaluator/report/{eval_id}")
async def get_report(eval_id: str):
    '''Proxy the report download request to Agent 2'''
    try:
        # First check if the report exists
        async with httpx.AsyncClient() as client:
            status_response = await client.get(f"{AGENT2_URL}/api/v1/evaluator/status/{eval_id}")
            status_data = status_response.json()
            
            if status_data.get("status") != "completed":
                raise HTTPException(status_code=404, detail="Report not ready yet")
                
            # Download the report
            report_response = await client.get(f"{AGENT2_URL}/api/v1/evaluator/report/{eval_id}")
            
            # If the response is a redirect to a file, handle it
            if report_response.status_code == 307:
                redirect_url = report_response.headers.get("location")
                if redirect_url:
                    report_response = await client.get(redirect_url)
            
            # Save the report temporarily and return it
            report_path = os.path.join(UPLOAD_DIR, f"report_{eval_id}.pdf")
            with open(report_path, "wb") as f:
                f.write(report_response.content)
                
            return FileResponse(
                path=report_path,
                filename=f"clustering_report_{eval_id}.pdf",
                media_type="application/pdf"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving report: {str(e)}")

# Get algorithm recommendations
@app.post("/api/get_recommendations")
async def get_recommendations(request: Request):
    '''Get algorithm recommendations based on data description'''
    try:
        data = await request.json()
        data_description = data.get("data_description", "")
        
        if not data_description:
            raise HTTPException(status_code=400, detail="Data description is required")
            
        query_rag = QueryOpenAi()
        recommendations = query_rag.get_algorithm_recommendations(data_description)
        
        return {"recommendations": json.loads(recommendations)}
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

# Streaming endpoint for AI chat assistant
@app.get("/stream")
async def stream(
    request: Request,
    search_query: str = Query(...),
    topNDocuments: int = Query(5),
    sessionID: str = Query(...),
):
    '''Provide streaming responses for chat interactions'''
    logger.info(
        f"search_query: {search_query}, topNDocuments: {topNDocuments}, sessionID: {sessionID}"
    )
    
    # Write sessionID to a file (from original implementation)
    with open("sessionID.txt", "w") as f:
        f.write(sessionID)

    query_rag = QueryOpenAi()
    
    # Modify the prompt to be clustering-aware
    enhanced_prompt = f"""
    You are an AI assistant specializing in data clustering and analysis. 
    The user is working with a clustering system that can perform various algorithms including 
    K-Means, DBSCAN, HDBSCAN, Gaussian Mixture Models, and Agglomerative Clustering.
    
    If the user seems to be asking about:
    - Data clustering concepts: Provide clear, concise explanations
    - Algorithm selection: Help them choose the most appropriate algorithm
    - Parameter tuning: Offer guidance on setting parameters
    - Result interpretation: Help them understand what their clustering results mean
    
    User query: {search_query}
    """

    def event_generator():
        response_chunks = []
        for content in query_rag.query_openai(enhanced_prompt):
            json_content = json.dumps({'type': 'response', 'data': content})
            # Make the response SSE compliant
            sse_content = f"data: {json_content}\n\n"
            yield sse_content

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)