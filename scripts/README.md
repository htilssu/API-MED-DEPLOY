# FAISS Index Download Script

This directory contains scripts for downloading FAISS index files during Docker container build time.

## Overview

The `download_faiss_indexes.py` script downloads required FAISS index files from Google Cloud Storage during the Docker build process, eliminating the need to download them at application startup.

## Files Downloaded

The script downloads the following files to `app/processed/`:
- `faiss_index.bin` - Main FAISS index for medical image similarity search
- `faiss_index_anomaly.bin` - FAISS index for anomaly detection
- `labels.npy` - Labels corresponding to the main FAISS index
- `labels_anomaly.npy` - Labels for anomaly detection index

## Usage

### Docker Build (Recommended)

The script is automatically executed during Docker build process. Make sure you have:

1. A valid `app/iam-key.json` Google Cloud service account key file
2. The service account has access to the `rag_3` bucket

```bash
docker build -t medical-api .
```

### Manual Execution

For testing or development purposes:

```bash
cd /path/to/project
python scripts/download_faiss_indexes.py
```

## Configuration

The script uses the following configuration:

- **GCS Bucket**: `rag_3`
- **GCS Folder**: `handle_data`
- **Local Directory**: `app/processed`
- **Credentials**: `app/iam-key.json` (with fallback to default GCP credentials)

## Error Handling

The script includes:
- Retry logic with exponential backoff (up to 5 attempts per file)
- File size verification after download
- Comprehensive logging
- Graceful fallback to default GCP credentials if IAM key is not found

## Authentication

Two authentication methods are supported:

1. **Service Account Key** (recommended for production):
   - Place your service account JSON key at `app/iam-key.json`

2. **Default Credentials** (for cloud environments):
   - Automatically used if no service account key is found
   - Works with Google Cloud Run, Compute Engine, etc.

## Troubleshooting

If downloads fail:

1. Check that `app/iam-key.json` exists and is valid
2. Verify the service account has `Storage Object Viewer` permissions on the `rag_3` bucket
3. Ensure network connectivity to Google Cloud Storage
4. Check the logs for specific error messages

The build will fail if any required files cannot be downloaded, ensuring the container always has complete data.