"""
Pydantic models for the Clustering Agent API.
Handles request validation, response formats, and data serialization.
"""

from pydantic import BaseModel, Field, validator
import json
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

class Hyperparameters(BaseModel):
    """Flexible hyperparameters model that accepts dict or JSON string"""
    params: Dict[str, Any] = Field(..., example={"n_clusters": 5})

    @validator('params', pre=True)
    def parse_hyperparameters(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle raw form data like "n_clusters=5"
                try:
                    return {k: int(v) if v.isdigit() else v 
                           for k,v in [pair.split('=') for pair in v.split('&')]}
                except:
                    raise ValueError("Invalid hyperparameters format")
        return v
class JobRequest(BaseModel):
    """Request model for clustering jobs with comprehensive validation."""
    data_url: Optional[str] = Field(
        None,
        example="https://example.com/data.csv",
        description="URL to fetch dataset (either this or file upload required)"
    )
    algorithm: str = Field(
        "kmeans",
        example="kmeans",
        description="Clustering algorithm to use",
        regex="^(kmeans|dbscan|hdbscan|gmm|agglomerative)$"
    )
    hyperparameters: Union[Dict[str, Any], Hyperparameters] = Field(
        {},
        description="Algorithm configuration parameters"
    )

    @validator('data_url')
    def validate_data_source(cls, v, values):
        if v is None and 'file' not in values:
            raise ValueError("Either data_url or file upload must be provided")
        return v

class JobResponse(BaseModel):
    """Standardized response for job initialization."""
    job_id: str = Field(
        ...,
        example="job_12345",
        description="Unique identifier for the clustering job"
    )
    status: str = Field(
        ...,
        example="started",
        description="Current job status",
        regex="^(created|started|in_progress|completed|failed)$"
    )

class ClusterMetrics(BaseModel):
    """Detailed model for clustering quality metrics."""
    silhouette_score: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Measure of cluster cohesion vs separation (-1 to 1)"
    )
    davies_bouldin_score: Optional[float] = Field(
        None,
        ge=0,
        description="Lower values indicate better clustering (≥0)"
    )
    calinski_harabasz_score: Optional[float] = Field(
        None,
        ge=0,
        description="Higher values indicate better clustering (≥0)"
    )
    inertia: Optional[float] = Field(
        None,
        ge=0,
        description="Sum of squared distances to nearest cluster center (≥0)"
    )
    aic: Optional[float] = Field(
        None,
        description="Akaike Information Criterion (GMM only)"
    )
    bic: Optional[float] = Field(
       