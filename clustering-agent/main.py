"""
This module provides a FastAPI application with endpoints for health check,
file upload, and running various clustering algorithms (K-Means, DBSCAN,
HDBSCAN, Gaussian Mixture Models) on the uploaded data.
"""

import logging
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Hyperparameters(BaseModel):
    """Model for hyperparameters."""
    params: Dict[str, float]

@app.get("/health")
def health_check():
    """Returns health status of the service."""
    return {"status": "healthy"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file and returns its content.

    Args:
        file (UploadFile): The uploaded file.

    Returns:
        dict: Filename and content of the file.
    """
    if file.content_type not in ["text/csv", "application/vnd.ms-excel",
                                 "text/plain", "text/markdown"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if file.content_type in ["text/csv", "application/vnd.ms-excel"]:
        df = pd.read_csv(file.file)
    elif file.content_type in ["text/plain", "text/markdown"]:
        df = pd.read_csv(file.file, delimiter="\t")
    else:
        df = pd.DataFrame()  # Ensure df is always defined

    return {"filename": file.filename, "content": df.head().to_dict()}

@app.post("/run_kmeans/")
async def run_kmeans(hyperparameters: Hyperparameters, file: UploadFile = File(...)):
    """
    Runs K-Means clustering on the uploaded file.

    Args:
        hyperparameters (Hyperparameters): Hyperparameters for K-Means.
        file (UploadFile): The uploaded file.

    Returns:
        dict: Labels and centroids of the clusters.
    """
    df = pd.read_csv(file.file)
    model = KMeans(**hyperparameters.params).fit(df)
    return {"labels": model.labels_.tolist(), "centroids": model.cluster_centers_.tolist()}

@app.post("/run_dbscan/")
async def run_dbscan(hyperparameters: Hyperparameters, file: UploadFile = File(...)):
    """
    Runs DBSCAN clustering on the uploaded file.

    Args:
        hyperparameters (Hyperparameters): Hyperparameters for DBSCAN.
        file (UploadFile): The uploaded file.

    Returns:
        dict: Labels of the clusters.
    """
    df = pd.read_csv(file.file)
    model = DBSCAN(**hyperparameters.params).fit(df)
    return {"labels": model.labels_.tolist()}

@app.post("/run_hdbscan/")
async def run_hdbscan(hyperparameters: Hyperparameters, file: UploadFile = File(...)):
    """
    Runs HDBSCAN clustering on the uploaded file.

    Args:
        hyperparameters (Hyperparameters): Hyperparameters for HDBSCAN.
        file (UploadFile): The uploaded file.

    Returns:
        dict: Labels of the clusters.
    """
    df = pd.read_csv(file.file)
    model = HDBSCAN(**hyperparameters.params).fit(df)
    return {"labels": model.labels_.tolist()}

@app.post("/run_gmm/")
async def run_gmm(hyperparameters: Hyperparameters, file: UploadFile = File(...)):
    """
    Runs Gaussian Mixture Model clustering on the uploaded file.

    Args:
        hyperparameters (Hyperparameters): Hyperparameters for GMM.
        file (UploadFile): The uploaded file.

    Returns:
        dict: Labels and means of the clusters.
    """
    df = pd.read_csv(file.file)
    model = GaussianMixture(**hyperparameters.params).fit(df)
    return {
        "labels": model.predict(df).tolist(),
        "means": model.means_.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
