#!/usr/bin/env python3
"""
Validation script to test if downloaded FAISS index files are working correctly.
This script verifies that the downloaded files can be loaded and used by the application.
"""

import os
import sys
import numpy as np

def validate_faiss_files():
    """Validate that FAISS index files are present and loadable."""
    print("🔍 Validating FAISS index files...")
    
    # Check if files exist
    files_to_check = [
        ("app/processed/faiss_index.bin", "Main FAISS index"),
        ("app/processed/faiss_index_anomaly.bin", "Anomaly FAISS index"),
        ("app/processed/labels.npy", "Main labels"),
        ("app/processed/labels_anomaly.npy", "Anomaly labels")
    ]
    
    missing_files = []
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {description}: {file_path} ({size} bytes)")
        else:
            print(f"❌ {description}: {file_path} (NOT FOUND)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files!")
        return False
    
    # Try to load the files
    print("\n🔍 Testing file loading...")
    try:
        # Test numpy files
        for file_path, description in [(f[0], f[1]) for f in files_to_check if f[0].endswith('.npy')]:
            labels = np.load(file_path, allow_pickle=True)
            print(f"✅ {description}: Loaded {len(labels)} labels")
        
        # Test FAISS files (if faiss is available)
        try:
            import faiss
            for file_path, description in [(f[0], f[1]) for f in files_to_check if f[0].endswith('.bin')]:
                index = faiss.read_index(file_path)
                print(f"✅ {description}: Loaded index with {index.ntotal} vectors, dimension {index.d}")
        except ImportError:
            print("⚠️  FAISS not installed - skipping FAISS index validation")
            print("   (This is OK if running outside the application container)")
        
        print("\n✅ All files validated successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ File validation failed: {e}")
        return False

def main():
    """Main validation function."""
    print("FAISS Files Validation Script")
    print("=" * 40)
    
    # Change to project directory if needed
    if not os.path.exists("app/processed"):
        print("⚠️  app/processed directory not found")
        print("   Make sure you're running this from the project root directory")
        print("   or that the FAISS files have been downloaded")
        return False
    
    success = validate_faiss_files()
    
    if success:
        print("\n🎉 Validation completed successfully!")
        print("   The FAISS index files are ready for use by the application.")
    else:
        print("\n💥 Validation failed!")
        print("   Please check the download process or file permissions.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)