#!/usr/bin/env python3
"""
Standalone script to download FAISS index files during Docker build.
This script downloads required FAISS indexes and labels from Google Cloud Storage
to be used by the medical diagnosis application.
"""

import os
import sys
import time
import logging
from pathlib import Path
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
GCS_BUCKET = "rag_3"
GCS_FOLDER = "handle_data"
LOCAL_SAVE_DIR = "app/processed"

# Files to download
REQUIRED_FILES = [
    "faiss_index.bin",
    "faiss_index_anomaly.bin",
    "labels.npy",
    "labels_anomaly.npy",
]

def get_gcs_client():
    """Initialize Google Cloud Storage client."""
    try:
        # Try to use service account key file first
        credentials_path = "app/iam-key.json"
        if os.path.exists(credentials_path):
            logger.info(f"Using service account credentials from {credentials_path}")
            return storage.Client.from_service_account_json(credentials_path)
        else:
            # Fall back to default credentials (useful for cloud environments)
            logger.info("IAM key file not found, trying default Google Cloud credentials")
            return storage.Client()
    except Exception as e:
        logger.error(f"Failed to create Google Cloud Storage client: {e}")
        logger.error("Make sure you have either:")
        logger.error("1. A valid app/iam-key.json service account key file, OR")
        logger.error("2. Default Google Cloud credentials configured in your environment")
        raise

def download_gcs_file(client, bucket_name, source_blob_name, destination_file_name, retries=5):
    """Download a file from Google Cloud Storage with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
            
            # Download with timeout
            blob.download_to_filename(destination_file_name, timeout=300)
            
            # Verify file was downloaded
            if os.path.exists(destination_file_name):
                file_size = os.path.getsize(destination_file_name)
                logger.info(f"Successfully downloaded {source_blob_name} → {destination_file_name} ({file_size} bytes)")
                return True
            else:
                raise Exception("File was not created after download")
                
        except Exception as e:
            logger.warning(f"Download attempt {attempt}/{retries} failed for {source_blob_name}: {e}")
            
            if attempt < retries:
                wait_time = 5 * attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {source_blob_name} after {retries} attempts")
                return False
    
    return False

def download_all_required_files():
    """Download all required FAISS index files."""
    logger.info("Starting FAISS index files download...")
    
    # Initialize GCS client
    try:
        client = get_gcs_client()
    except Exception as e:
        logger.error(f"Cannot initialize GCS client: {e}")
        return False
    
    # Create target directory
    Path(LOCAL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created target directory: {LOCAL_SAVE_DIR}")
    
    # Download each required file
    success_count = 0
    total_files = len(REQUIRED_FILES)
    
    for file in REQUIRED_FILES:
        gcs_path = f"{GCS_FOLDER}/{file}"
        local_path = os.path.join(LOCAL_SAVE_DIR, file)
        
        logger.info(f"Downloading {file}... ({success_count + 1}/{total_files})")
        
        if download_gcs_file(client, GCS_BUCKET, gcs_path, local_path):
            success_count += 1
        else:
            logger.error(f"Failed to download required file: {file}")
    
    # Check if all files were downloaded successfully
    if success_count == total_files:
        logger.info(f"Successfully downloaded all {total_files} FAISS index files")
        
        # List downloaded files with sizes
        logger.info("Downloaded files:")
        for file in REQUIRED_FILES:
            local_path = os.path.join(LOCAL_SAVE_DIR, file)
            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                logger.info(f"  - {file}: {size} bytes")
        
        return True
    else:
        logger.error(f"Only {success_count}/{total_files} files downloaded successfully")
        return False

def main():
    """Main function."""
    logger.info("FAISS Index Download Script - Starting")
    
    try:
        success = download_all_required_files()
        
        if success:
            logger.info("FAISS index download completed successfully")
            sys.exit(0)
        else:
            logger.error("FAISS index download failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()