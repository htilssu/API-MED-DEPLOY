# FAISS Index Download Scripts

This directory contains scripts for downloading and validating FAISS index files during Docker container build time.

## Overview

The scripts in this directory enable downloading required FAISS index files from Google Cloud Storage during the Docker build process, eliminating the need to download them at application startup.

## Scripts

### `download_faiss_indexes.py`
Main download script that fetches FAISS index files from GCS.

### `validate_faiss_files.py`
Validation script to verify downloaded files are present and loadable.

## Files Downloaded

The download script fetches the following files to `app/processed/`:
- `faiss_index.bin` - Main FAISS index for medical image similarity search
- `faiss_index_anomaly.bin` - FAISS index for anomaly detection
- `labels.npy` - Labels corresponding to the main FAISS index
- `labels_anomaly.npy` - Labels for anomaly detection index

## Usage

### Docker Build (Recommended)

The download script is automatically executed during Docker build process:

```bash
# Ensure you have app/iam-key.json with proper GCS permissions
docker build -t medical-api .
```

### Manual Download

For testing or development purposes:

```bash
cd /path/to/project
python scripts/download_faiss_indexes.py
```

### Validation

To verify downloaded files are working correctly:

```bash
python scripts/validate_faiss_files.py
```

## Configuration

The download script uses the following configuration:

- **GCS Bucket**: `rag_3`
- **GCS Folder**: `handle_data`
- **Local Directory**: `app/processed`
- **Credentials**: `app/iam-key.json` (with fallback to default GCP credentials)

## Error Handling

The download script includes:
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
5. Run the validation script to verify file integrity

The build will fail if any required files cannot be downloaded, ensuring the container always has complete data.

## Docker Build Process

The updated Dockerfile uses a 3-stage build:

1. **Stage 1 (builder)**: Install Python dependencies
2. **Stage 2 (downloader)**: Download FAISS indexes
3. **Stage 3 (runtime)**: Create final image with pre-downloaded files

This approach ensures:
- Fast container startup (no download delay)
- Reliable deployments (fails fast if files unavailable)
- Reduced runtime dependencies
- Better container immutability