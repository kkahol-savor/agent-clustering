"""
PDF report generation for clustering analysis results.
"""

import os
import json
import base64
import logging
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Dict, Any, List, Optional
from fpdf import FPDF
import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ClusteringReport(FPDF):
    """Custom PDF class for clustering reports."""
    
    def __init__(self):
        super().__init__()
        self.width = 210  # A4 width in mm
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", "", 12)
        
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
        
    def add_image(self, image_data, caption):
        """Add an image with caption."""
        self.image(image_data, x=10, y=None, w=190)
        self.set_font("Arial", "I", 10)
        self.cell(0, 5, caption, 0, 1, "C")
        self.ln(5)

async def generate_report(
    eval_id: str,
    job_id: str,
    evaluation_results: Dict[str, Any],
    optimized: bool,
    best_params: Optional[Dict[str, Any]],
    reports_dir: str
) -> str:
    """
    Generate a PDF report for clustering analysis.
    
    Args:
        eval_id: Evaluation ID
        job_id: Clustering job ID
        evaluation_results: Results of the evaluation
        optimized: Whether hyperparameter optimization was performed
        best_params: Best parameters from optimization (if performed)
        reports_dir: Directory to save the report
        
    Returns:
        report_path: Path to the generated PDF report
    """
    try:
        # Create PDF
        pdf = ClusteringReport()
        
        # Executive Summary
        pdf.chapter_title("Executive Summary")
        
        algorithm = evaluation_results.get("algorithm", "Unknown")
        metrics_analysis = evaluation_results.get("metrics_analysis", {})
        clusters_analysis = evaluation_results.get("clusters_analysis", {})
        
        summary = (
            f"This report analyzes the results of {algorithm} clustering performed on your data. "
            f"The analysis identified {clusters_analysis.get('num_clusters', 0)} clusters "
            f"from {clusters_analysis.get('total_points', 0)} data points.\n\n"
        )
        
        # Add quality assessment based on silhouette score if available
        if "silhouette_interpretation" in metrics_analysis:
            quality = metrics_analysis["silhouette_interpretation"]["quality"]
            summary += f"The clustering quality is assessed as {quality.upper()} based on the silhouette score.\n\n"
            
        # Add noise assessment if available
        if "noise_analysis" in clusters_analysis:
            noise = clusters_analysis["noise_analysis"]
            summary += f"The clustering identified {noise['percentage']:.1f}% of points as noise or outliers.\n\n"
            
        # Add optimization summary if performed
        if optimized and best_params:
            summary += (
                f"Hyperparameter optimization was performed to improve clustering quality. "
                f"The optimal parameters were identified and used for the final clustering.\n\n"
            )
            
        pdf.chapter_body(summary)
        
        # Clustering Configuration
        pdf.chapter_title("Clustering Configuration")
        
        config_text = f"Algorithm: {algorithm}\n\n"
        
        # Original parameters would be ideal here, but we may not have them
        # Add optimized parameters if available
        if best_params:
            config_text += "Optimal Parameters:\n"
            for param, value in best_params.items():
                config_text += f"- {param}: {value}\n"
        
        pdf.chapter_body(config_text)
        
        # Cluster Quality
        pdf.chapter_title("Cluster Quality Assessment")
        
        quality_text = ""
        
        # Add metrics interpretations
        for metric_name, analysis in metrics_analysis.items():
            if isinstance(analysis, dict) and "explanation" in analysis:
                quality_text += f"{analysis['explanation']}\n\n"
        
        pdf.chapter_body(quality_text)
        
        # Cluster Distribution
        pdf.chapter_title("Cluster Distribution")
        
        distribution_text = f"Number of Clusters: {clusters_analysis.get('num_clusters', 0)}\n"
        distribution_text += f"Total Data Points: {clusters_analysis.get('total_points', 0)}\n\n"
        
        # Add cluster sizes
        if "cluster_distribution" in clusters_analysis:
            distribution_text += "Cluster Sizes:\n"
            for label, info in clusters_analysis["cluster_distribution"].items():
                cluster_name = "Noise" if label == "-1" or label == "-1.0" else f"Cluster {label}"
                distribution_text += f"- {cluster_name}: {info['count']} points ({info['percentage']:.1f}%)\n"
        
        # Add balance analysis if available
        if "balance_analysis" in clusters_analysis:
            balance = clusters_analysis["balance_analysis"]
            distribution_text += f"\nCluster Balance Assessment: {balance['assessment']}\n"
            distribution_text += f"{balance['impact']}\n"
            
        pdf.chapter_body(distribution_text)
        
        # Create a simple visualization of cluster distribution
        if "cluster_distribution" in clusters_analysis:
            plt.figure(figsize=(10, 6))
            
            # Prepare data for plotting
            labels = []
            sizes = []
            for label, info in clusters_analysis["cluster_distribution"].items():
                cluster_name = "Noise" if label == "-1" or label == "-1.0" else f"Cluster {label}"
                labels.append(cluster_name)
                sizes.append(info["count"])
                
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Cluster Distribution')
            
            # Save the plot to a BytesIO object
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plt.close()
            
            # Add the plot to the PDF
            pdf.add_image(img_buffer, "Cluster Size Distribution")
        
        # Optimization Results (if applicable)
        if optimized and best_params:
            pdf.chapter_title("Hyperparameter Optimization")
            
            opt_text = "The following parameters were found to produce optimal clustering results:\n\n"
            
            for param, value in best_params.items():
                opt_text += f"{param}: {value}\n"
                
            if "optimized" in evaluation_results and "metrics" in evaluation_results["optimized"]:
                opt_metrics = evaluation_results["optimized"]["metrics"]
                
                opt_text += "\nMetrics with Optimized Parameters:\n"
                for metric, value in opt_metrics.items():
                    opt_text += f"- {metric}: {value}\n"
                    
            pdf.chapter_body(opt_text)
        
        # Conclusions & Next Steps
        pdf.chapter_title("Conclusions & Next Steps")
        
        conclusions = "Based on the clustering analysis, we recommend the following next steps:\n\n"
        
        # Quality-based recommendations
        quality_level = "high"
        if "silhouette_interpretation" in metrics_analysis:
            quality = metrics_analysis["silhouette_interpretation"]["quality"]
            if quality in ["poor", "fair"]:
                quality_level = "low"
                conclusions += (
                    "1. The clustering quality is suboptimal. Consider the following improvements:\n"
                    "   - Try different clustering algorithms\n"
                    "   - Further optimize hyperparameters\n"
                    "   - Review and preprocess the data to remove noise or outliers\n"
                    "   - Consider dimensionality reduction techniques\n\n"
                )
            else:
                conclusions += (
                    "1. The clustering quality is good. You can proceed with using these clusters for your application.\n\n"
                )
        
        # Noise-based recommendations
        if "noise_analysis" in clusters_analysis:
            noise_level = clusters_analysis["noise_analysis"]["assessment"]
            if noise_level in ["High", "Very high"]:
                conclusions += (
                    "2. The high level of noise suggests:\n"
                    "   - Your data may contain many outliers\n"
                    "   - The chosen algorithm may not be suitable for your data structure\n"
                    "   - Consider preprocessing steps to clean the data\n\n"
                )
                
        # Balance-based recommendations
        if "balance_analysis" in clusters_analysis:
            balance_level = clusters_analysis["balance_analysis"]["assessment"]
            if "imbalanced" in balance_level.lower():
                conclusions += (
                    "3. The clusters are imbalanced in size. This may be natural for your data, but consider:\n"
                    "   - If this matches your domain knowledge\n"
                    "   - Whether different hyperparameters could produce more balanced clusters\n"
                    "   - Whether the largest cluster should be further subdivided\n\n"
                )
                
        # General next steps
        if quality_level == "high":
            conclusions += (
                "4. Recommended next steps with your quality clusters:\n"
                "   - Analyze the characteristics of each cluster\n"
                "   - Use the clusters for downstream tasks such as targeted marketing or segmentation\n"
                "   - Consider visualizing the clusters in lower dimensions using PCA or t-SNE\n"
            )
        else:
            conclusions += (
                "4. To improve clustering results:\n"
                "   - Try additional algorithms beyond " + algorithm + "\n"
                "   - Consider feature engineering to create more discriminative features\n"
                "   - Investigate domain-specific transformations of your data\n"
            )
                
        pdf.chapter_body(conclusions)
        
        # Save the PDF
        report_path = os.path.join(reports_dir, f"clustering_report_{eval_id}.pdf")
        pdf.output(report_path)
        
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise Exception(f"Failed to generate PDF report: {str(e)}")