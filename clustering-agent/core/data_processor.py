"""
Data processing functionality for the clustering agent,
handling file uploads, downloads, and loading.
"""

import os
import uuid
import aiohttp
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from fastapi import UploadFile
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data directory
DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

class DataProcessor:
    """
    Handles data processing operations such as file uploads,
    downloads, and loading data for clustering.
    """
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """
        Save an uploaded file to disk.
        
        Args:
            file: The uploaded file
            
        Returns:
            file_path: Path to the saved file
        """
        # Make sure file has an extension
        file_ext = os.path.splitext(file.filename)[1]
        if not file_ext:  # If no extension, assume CSV
            file_ext = ".csv"
            
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(DATA_DIR, file_name)
        
        logger.info(f"Saving file {file.filename} to {file_path}")
        
        # Read file content
        contents = await file.read()
        
        # Write to file
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # Log file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved: {file_path}, Size: {file_size} bytes")
            
        return file_path
        
    async def download_file(self, url: str) -> str:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download from
            
        Returns:
            file_path: Path to the downloaded file
        """
        # Generate a unique filename
        file_name = f"{uuid.uuid4()}.csv"  # Assume CSV for now
        file_path = os.path.join(DATA_DIR, file_name)
        
        logger.info(f"Downloading file from {url} to {file_path}")
        
        # Download the file
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download file from {url}: {response.status}")
                    raise Exception(f"Failed to download file from {url}: {response.status}")
                    
                # Read response and write to file
                data = await response.read()
                with open(file_path, "wb") as f:
                    f.write(data)
                    
                # Log file size
                file_size = os.path.getsize(file_path)
                logger.info(f"File downloaded: {file_path}, Size: {file_size} bytes")
                    
        return file_path
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file into a pandas DataFrame.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the loaded data
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"Loading data from {file_path} with extension {file_ext}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Read first few bytes to diagnose file type/encoding issues
        with open(file_path, 'rb') as f:
            file_start = f.read(100)
            logger.info(f"File starts with: {file_start}")
        
        try:
            # Try different loading methods based on file extension
            if file_ext == ".csv":
                # Try with different potential separators and encodings
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"Successfully loaded CSV with default settings: {len(df)} rows")
                except Exception as e1:
                    logger.warning(f"Failed to load CSV with default settings: {str(e1)}")
                    try:
                        # Try with explicit separator
                        df = pd.read_csv(file_path, sep=',')
                        logger.info(f"Successfully loaded CSV with comma separator: {len(df)} rows")
                    except Exception as e2:
                        logger.warning(f"Failed with comma separator: {str(e2)}")
                        try:
                            # Try with tab separator
                            df = pd.read_csv(file_path, sep='\t')
                            logger.info(f"Successfully loaded CSV with tab separator: {len(df)} rows")
                        except Exception as e3:
                            logger.warning(f"Failed with tab separator: {str(e3)}")
                            try:
                                # Try with different encoding
                                df = pd.read_csv(file_path, encoding='latin1')
                                logger.info(f"Successfully loaded CSV with latin1 encoding: {len(df)} rows")
                            except Exception as e4:
                                logger.error(f"All CSV loading attempts failed")
                                # If all else fails, try a very permissive read
                                df = pd.read_csv(file_path, sep=None, engine='python')
                                logger.info(f"Loaded with auto-detected separator: {len(df)} rows")
            
            elif file_ext in [".json", ".jsonl"]:
                df = pd.read_json(file_path)
                logger.info(f"Successfully loaded JSON: {len(df)} rows")
            
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                logger.info(f"Successfully loaded Excel: {len(df)} rows")
            
            else:
                logger.warning(f"Unrecognized extension: {file_ext}, trying CSV format")
                df = pd.read_csv(file_path)
                logger.info(f"Successfully loaded as CSV: {len(df)} rows")
            
            # Check for minimum data requirements
            if len(df) < 2:
                logger.error(f"Not enough data: {len(df)} rows")
                raise ValueError(f"File contains only {len(df)} rows, minimum 2 rows required")
                
            # Convert string numbers to numeric
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        df[col] = pd.to_numeric(df[col])
                        logger.info(f"Converted column {col} to numeric")
                    except ValueError:
                        logger.info(f"Column {col} remains non-numeric")
            
            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 1:
                logger.error("No numeric columns found in data")
                raise ValueError("No numeric columns found. At least one numeric column is required for clustering.")
            
            logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Numeric columns: {numeric_cols}")
            
            # Check for missing values
            missing_values = df[numeric_cols].isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"Data contains {missing_values} missing values in numeric columns. These will be handled during processing.")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Read file content for debug purposes (limited to first 100 bytes)
            with open(file_path, 'rb') as f:
                content_preview = f.read(100)
            logger.error(f"File content preview: {content_preview}")
            raise ValueError(f"Could not process file: {str(e)}")
            
    def preview_data(self, file_path: str, max_rows: int = 5) -> Dict[str, Any]:
        """
        Generate a preview of the data for inspection.
        
        Args:
            file_path: Path to the data file
            max_rows: Maximum number of rows to preview
            
        Returns:
            Dictionary with data preview information
        """
        try:
            df = self.load_data(file_path)
            
            # Get basic statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            preview = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "non_numeric_columns": [col for col in df.columns if col not in numeric_cols],
                "sample_rows": df.head(max_rows).to_dict(orient="records"),
                "column_types": {col: str(df[col].dtype) for col in df.columns}
            }
            
            # Add basic stats for numeric columns
            if numeric_cols:
                preview["numeric_stats"] = {}
                for col in numeric_cols:
                    preview["numeric_stats"][col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "missing": int(df[col].isnull().sum())
                    }
            
            return preview
            
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            return {
                "error": str(e),
                "file_path": file_path
            }