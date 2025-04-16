"""
Pydantic models for the Evaluator & Optimizer Agent API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class NotifySMS(BaseModel):
    """Settings for SMS notifications."""
    enabled: bool = False
    phone_number: Optional[str] = None
    carrier_gateway: Optional[str] = None

class OptimizationConfig(BaseModel):
    """Configuration for hyperparameter optimization."""
    max_iterations: int = 10
    search_algorithm: str = "optuna"  # or "hyperopt", "random", etc.

class EvaluationRequest(BaseModel):
    """Request model for clustering evaluation."""
    job_id: str
    notify_sms: Optional[NotifySMS] = None
    optimize: bool = False
    optimization_config: Optional[OptimizationConfig] = None

class EvaluationResponse(BaseModel):
    """Response model for evaluation initiation."""
    eval_id: str
    status: str

class StatusResponse(BaseModel):
    """Response model for evaluation status."""
    eval_id: str
    status: str
    best_params: Optional[Dict[str, Any]] = None