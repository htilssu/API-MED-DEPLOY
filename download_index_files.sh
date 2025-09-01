#!/bin/bash

# Script để download các file cần thiết từ Google Cloud Storage
# Tạo bởi: GitHub Copilot  
# Mục đích: Download faiss index và label files vào folder index
#
# CÁCH SỬ DỤNG:
# ./download_index_files.sh
#
# FILES SẼ ĐƯỢC DOWNLOAD:
# - faiss_index.bin
# - faiss_index_anomaly.bin  
# - labels_anomaly.npy
# - labels.npy
#
# REQUIREMENTS:
# Script này yêu cầu quyền truy cập vào bucket 'demer' trên Google Cloud Storage
# Để sử dụng gsutil, bạn cần:
# 1. Cài đặt và cấu hình Google Cloud SDK
# 2. Chạy: gcloud auth login
# 3. Hoặc thiết lập service account credentials
#
# FALLBACK: Script sẽ thử các phương thức sau theo thứ tự:
# 1. gsutil (authenticated access)
# 2. wget (public access)  
# 3. curl (public access)

set -e  # Exit on any error

# Định nghĩa bucket và files
BUCKET="demer"
FILES=(
    "faiss_index.bin"
    "faiss_index_anomaly.bin"
    "labels_anomaly.npy"
    "labels.npy"
)

# Tạo folder index nếu chưa tồn tại
INDEX_DIR="index"
echo "Tạo folder ${INDEX_DIR}..."
mkdir -p "${INDEX_DIR}"

# Function để download file với gsutil
download_file_gsutil() {
    local filename="$1"
    local gs_path="gs://${BUCKET}/${filename}"
    local output_path="${INDEX_DIR}/${filename}"
    
    echo "Đang download ${filename} bằng gsutil..."
    
    # Sử dụng timeout để tránh hang
    if timeout 30 gsutil -q cp "${gs_path}" "${output_path}" 2>/dev/null; then
        echo "✓ Download thành công: ${filename}"
        return 0
    else
        echo "✗ Lỗi download ${filename} với gsutil (có thể do authentication hoặc quyền truy cập)"
        return 1
    fi
}

# Function để download file với wget (public URLs)
download_file_wget() {
    local filename="$1"
    local url="https://storage.googleapis.com/${BUCKET}/${filename}"
    local output_path="${INDEX_DIR}/${filename}"
    
    echo "Đang download ${filename} bằng wget..."
    
    if wget -q --timeout=30 -O "${output_path}" "${url}" 2>/dev/null; then
        # Kiểm tra nếu file có nội dung (không phải error page)
        if [ -s "${output_path}" ]; then
            # Kiểm tra nếu file chứa error message thường gặp
            if grep -q -i "blocked\|denied\|forbidden\|error\|not found" "${output_path}" 2>/dev/null; then
                rm -f "${output_path}"
                echo "✗ File chứa error message thay vì dữ liệu thực"
            else
                echo "✓ Download thành công: ${filename}"
                return 0
            fi
        else
            rm -f "${output_path}"  # Xóa file rỗng
        fi
    fi
    
    echo "✗ Lỗi download ${filename} với wget (có thể file không public)"
    return 1
}

# Function để download file (thử cả gsutil và wget)
download_file() {
    local filename="$1"
    
    # Thử gsutil trước (cho authenticated access)
    if command -v gsutil >/dev/null 2>&1; then
        if download_file_gsutil "$filename"; then
            return 0
        fi
    fi
    
    # Nếu gsutil thất bại hoặc không có, thử wget
    if command -v wget >/dev/null 2>&1; then
        if download_file_wget "$filename"; then
            return 0
        fi
    fi
    
    # Nếu cả hai đều thất bại, thử curl
    if command -v curl >/dev/null 2>&1; then
        echo "Đang download ${filename} bằng curl..."
        local url="https://storage.googleapis.com/${BUCKET}/${filename}"
        local output_path="${INDEX_DIR}/${filename}"
        if curl -L -s --max-time 30 -o "${output_path}" "${url}" 2>/dev/null; then
            # Kiểm tra xem file có được download thành công không và không phải error page
            if [ -s "${output_path}" ]; then
                # Kiểm tra nếu file chứa error message thường gặp
                if grep -q -i "blocked\|denied\|forbidden\|error\|not found" "${output_path}" 2>/dev/null; then
                    rm -f "${output_path}"
                    echo "✗ File chứa error message thay vì dữ liệu thực"
                else
                    echo "✓ Download thành công: ${filename}"
                    return 0
                fi
            else
                rm -f "${output_path}"  # Xóa file rỗng
            fi
        fi
        echo "✗ Lỗi download ${filename} với curl"
    fi
    
    echo "✗ Thất bại download ${filename} với tất cả các phương thức"
    return 1
}

# Main execution
echo "=== Bắt đầu download files từ Google Cloud Storage ==="
echo "Destination: ${INDEX_DIR}/"
echo ""

success_count=0
total_files=${#FILES[@]}

for file in "${FILES[@]}"; do
    if download_file "$file"; then
        success_count=$((success_count + 1))
    fi
    echo ""
done

echo "=== Kết quả ==="
echo "Thành công: ${success_count}/${total_files} files"

if [ $success_count -eq $total_files ]; then
    echo "✓ Tất cả files đã được download thành công!"
    echo ""
    echo "Files trong folder ${INDEX_DIR}:"
    ls -la "${INDEX_DIR}/"
    exit 0
else
    echo "✗ Một số files không thể download."
    echo ""
    echo "Có thể do các lý do sau:"
    echo "1. Files không public và yêu cầu authentication"
    echo "2. Cần cấu hình Google Cloud credentials:"
    echo "   - Chạy: gcloud auth login"
    echo "   - Hoặc thiết lập GOOGLE_APPLICATION_CREDENTIALS"
    echo "3. Không có quyền truy cập bucket 'demer'"
    echo ""
    echo "Vui lòng kiểm tra authentication và thử lại."
    exit 1
fi