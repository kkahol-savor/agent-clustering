 Technical Specification: Clustering and Evaluation Agents
 1. Introduction
 This document outlines the technical specifications for a two-agent system designed to perform automated clustering analysis, iterative evaluation, hyperparameter optimization, scheduled execution, PDF report generation, and SMS notifications using SendGrid.
 The system consists of two components deployed as independent FastAPI services, which can be deployed on the same machine or on separate infrastructures:
 1.	Agent 1: Clustering Agent
 o	Runs various clustering algorithms (K-Means, DBSCAN, HDBSCAN, Gaussian Mixture Models, etc.).
 o	Accepts hyperparameters, data, and clustering method requests.
 o	Exposes results via a REST API.
 2.	Agent 2: Evaluator & Optimizer Agent
 o	Evaluates the clustering results from Agent 1 using multiple metrics.
 o	Suggests/executes hyperparameter optimization steps and re-runs Agent 1 for improved clustering.
 o	Generates comprehensive PDF reports.
 o	Sends SMS notifications via SendGrid upon completion.
 o	Can schedule regular or repeated clustering tasks using APScheduler.
  
 2. High-Level Architecture
 java
 CopyEdit
 ┌─────────────────────────┐
 │ Agent 1: Clustering API │
 │   (FastAPI Service)     │
 └─────────┬───────────────┘
           │
    Clustering Results (Labels, Centroids, etc.)
           │
           ▼
 ┌───────────────────────────────────┐
 │ Agent 2: Evaluator & Optimizer   │
 │ (FastAPI Service + APScheduler)  │
 └───────────────────────────────────┘
           │
           ├─ Evaluate Clusters (Silhouette, CH, DBI, etc.)
           ├─ Optimize Hyperparameters (Optuna/Hyperopt)
           ├─ Generate PDF Report
           └─ Send SMS Notifications via SendGrid
 2.1 Deployment Considerations
 •	Both agents can be containerized using Docker and orchestrated via Docker Compose, Kubernetes, or other container platforms.
 •	Communication between Agent 1 and Agent 2 occurs via REST calls.
 •	The system can be integrated with an external scheduler (e.g., Cron, Airflow) or rely on APScheduler in Agent 2 for internal scheduling.
  
 3. Agent 1: Clustering Service
 3.1 Overview
 Agent 1 is responsible for receiving input data (by URL or directly via request body), clustering that data with specified algorithms and hyperparameters, and returning status and results.
 3.2 Base URL
 bash
 CopyEdit
 /api/v1/clustering
 3.3 Endpoints
 1.	POST /run
 o	Description: Starts a clustering job based on request parameters (algorithm, hyperparameters, etc.).
 o	Request Body (JSON):
 json
 CopyEdit
 {
   "data_url": "https://example.com/data.csv",
   "algorithm": "kmeans",
   "hyperparameters": {
     "n_clusters": 5,
     "init": "k-means++"
   }
 }
 	data_url or a direct data payload containing features (CSV, JSON, etc.).
 	algorithm: Identifies which algorithm to use (kmeans, dbscan, hdbscan, gmm, etc.).
 	hyperparameters: Key-value pairs for algorithm configuration.
 o	Response (JSON):
 json
 CopyEdit
 {
   "job_id": "job_12345",
   "status": "started"
 }
 2.	GET /status/{job_id}
 o	Description: Checks the status of a running or completed clustering job.
 o	Response (JSON): 
 json
 CopyEdit
 {
   "job_id": "job_12345",
   "status": "completed" 
   // or "in_progress", "error"
 }
 3.	GET /results/{job_id}
 o	Description: Retrieves the final clustering results for a completed job.
 o	Response (JSON): 
 json
 CopyEdit
 {
   "job_id": "job_12345",
   "status": "completed",
   "results": {
     "labels": [...],
     "centroids": [...],
     "metrics": {
       "silhouette_score": 0.65,
       "davies_bouldin": 0.9
     }
   }
 }
 3.4 Supported Algorithms
 •	K-Means (KMeans from scikit-learn)
 •	DBSCAN (DBSCAN from scikit-learn)
 •	HDBSCAN (from hdbscan library)
 •	Gaussian Mixture Models (GaussianMixture from scikit-learn)
 •	Agglomerative Clustering (AgglomerativeClustering from scikit-learn)
 3.5 Input Data & Hyperparameters
 •	Data Formats: CSV, JSON, or direct array-like objects in request.
 •	Hyperparameters vary by algorithm. Examples: 
 o	K-Means: n_clusters, init, max_iter, n_init
 o	DBSCAN: eps, min_samples
 o	HDBSCAN: min_cluster_size, min_samples
 o	GaussianMixture: n_components, covariance_type
 o	Agglomerative: n_clusters, linkage (ward, complete, average, etc.)
  
 4. Agent 2: Evaluator & Optimizer Service
 4.1 Overview
 Agent 2 evaluates clustering results from Agent 1, optimizing hyperparameters iteratively. It can schedule repeated runs (daily, weekly, etc.), generate PDF reports, and notify users via SMS using SendGrid.
 4.2 Base URL
 bash
 CopyEdit
 /api/v1/evaluator
 4.3 Endpoints
 1.	POST /evaluate
 o	Description: Initiates a clustering evaluation. Retrieves the results from Agent 1, runs evaluation metrics, can re-run Agent 1 with new hyperparameters if desired, and optionally generates a PDF report.
 o	Request Body (JSON):
 json
 CopyEdit
 {
   "job_id": "job_12345",
   "notify_sms": {
     "enabled": true,
     "phone_number": "+1234567890",
     "carrier_gateway": "vtext.com"
   },
   "optimize": true,
   "optimization_config": {
     "max_iterations": 10,
     "search_algorithm": "optuna"
   }
 }
 	job_id: ID of the clustering job from Agent 1.
 	notify_sms: If enabled, sends SMS upon evaluation/report completion.
 	optimize: If true, triggers hyperparameter tuning.
 	optimization_config: Settings for the hyperparameter optimization process (e.g., iteration limit, library choice).
 o	Response (JSON):
 json
 CopyEdit
 {
   "eval_id": "eval_67890",
   "status": "evaluation_started"
 }
 2.	GET /status/{eval_id}
 o	Description: Checks the status of an ongoing or completed evaluation.
 o	Response (JSON): 
 json
 CopyEdit
 {
   "eval_id": "eval_67890",
   "status": "completed", 
   "best_params": {...}
 }
 3.	GET /report/{eval_id}
 o	Description: Retrieves the PDF report generated after the evaluation.
 o	Response: Returns a binary PDF file or a download link.
 4.4 Evaluation Metrics
 •	Silhouette Score
 •	Calinski-Harabasz Index
 •	Davies-Bouldin Index
 •	Inertia (where applicable, e.g., K-Means)
 •	Cluster Stability or Homogeneity (optional, depending on approach)
 4.5 Hyperparameter Optimization
 •	Libraries: Optuna, Hyperopt, or sklearn.model_selection strategies.
 •	Method: 
 1.	Start with an initial set of hyperparameters.
 2.	Evaluate performance using the chosen metric(s).
 3.	Update hyperparameters (Bayesian, random search, TPE, etc.).
 4.	Re-run Agent 1.
 5.	Repeat until a stopping criterion is met (max_iterations or convergence).
 4.6 Scheduling
 •	APScheduler in Agent 2 handles periodic or ad-hoc scheduling: 
 python
 CopyEdit
 from apscheduler.schedulers.background import BackgroundScheduler
 
 scheduler = BackgroundScheduler()
 scheduler.add_job(
     func=run_evaluation_pipeline, 
     trigger='interval', 
     days=1,
     next_run_time=datetime.now()
 )
 scheduler.start()
 o	run_evaluation_pipeline can call Agent 1’s /run endpoint, then call /evaluate after completion.
  
 5. PDF Report Generation
 5.1 Contents
 •	Executive Summary
 •	Data Overview (size, feature distribution)
 •	Clustering Configuration (algorithm, hyperparameters, iteration logs)
 •	Cluster Quality (Silhouette, Davies-Bouldin, etc.)
 •	Visualizations: 
 o	2D/3D PCA or t-SNE plots with cluster coloring
 o	Distribution histograms per cluster
 •	Best Hyperparameters (if optimization is performed)
 •	Conclusions & Next Steps
 5.2 Libraries
 •	FPDF or ReportLab for generating PDF files programmatically.
 •	Matplotlib for creating plots, saved as images and embedded in PDF.
  
 6. SMS Notifications via SendGrid
 6.1 Overview
 •	SendGrid typically sends emails. For SMS, Agent 2 leverages carrier email gateways (e.g., vtext.com for Verizon).
 •	Example (Python snippet): 
 python
 CopyEdit
 import sendgrid
 from sendgrid.helpers.mail import Mail
 
 def send_sms_via_sendgrid(phone_number, gateway, message):
     sg = sendgrid.SendGridAPIClient(api_key='YOUR_API_KEY')
     to_email = f"{phone_number}@{gateway}"
     mail = Mail(
         from_email='noreply@domain.com',
         to_emails=to_email,
         subject='Clustering Evaluation Complete',
         plain_text_content=message
     )
     response = sg.send(mail)
     return response.status_code
 •	Message Example:
 “Clustering evaluation complete. Report: https://example.com/api/v1/evaluator/report/eval_67890”
  
 7. Technology Stack
 Component	Technology
 API Framework	FastAPI
 Clustering & ML	scikit-learn, hdbscan
 Hyperparameter Optimization	Optuna, Hyperopt
 Scheduling	APScheduler (integrated into Agent 2)
 Containerization	Docker, Docker Compose (optional)
 PDF Reports	FPDF, ReportLab
 SMS Notifications	SendGrid (email-to-SMS)
 Deployment	Any standard server / container platform
  
 8. Security and Authentication
 8.1 API Security
 •	Token-based authentication (e.g., OAuth2, JWT) recommended for both agents, especially if deployed publicly.
 •	HTTPS for secure data transfer.
 8.2 Data Privacy
 •	Ensure sensitive data (e.g., phone numbers) is encrypted or stored securely.
 •	Compliance with relevant data protection regulations (GDPR, HIPAA, etc.) if applicable.
  
 9. Error Handling and Logging
 9.1 Error Responses
 •	Agent 1 returns meaningful errors for invalid hyperparameters or invalid data format.
 •	Agent 2 returns errors for unrecognized evaluation IDs, job IDs, or scheduling issues.
 9.2 Logging
 •	Structured logging (JSON logs) recommended for easy aggregation.
 •	Log cluster results, evaluation metrics, and notifications.
  
 10. Next Steps
 1.	Implement FastAPI Services for Agent 1 & Agent 2 with the described endpoints.
 2.	Configure APScheduler in Agent 2 for scheduled tasks.
 3.	Integrate SendGrid to dispatch SMS notifications upon evaluation completion.
 4.	Develop PDF report generation (FPDF, ReportLab) with the specified structure.
 5.	Containerize (Docker) and configure environment variables for secrets (API keys).
 6.	Test end-to-end flow: 
 1.	Agent 2 schedules a run.
 2.	Agent 1 does clustering.
 3.	Agent 2 evaluates results, possibly iterates.
 4.	Agent 2 generates a PDF and sends an SMS notification.
  
 Document Revision History
 Version	Date	Author	Change
 1.0	[March 11 2025]	[Kanav Kahol]	Initial Technical Specification
 
 