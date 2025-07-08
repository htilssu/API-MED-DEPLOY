from pathlib import Path
import os

IMAGE_DIR = "app/static/images/dataset"

def count_images():
    # Định nghĩa các phần mở rộng ảnh
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    
    # Tạo đối tượng Path
    image_dir = Path(IMAGE_DIR)
    
    # Kiểm tra xem thư mục tồn tại
    if not image_dir.exists():
        print(f"Thư mục {IMAGE_DIR} không tồn tại!")
        return
    
    # Đếm tổng số ảnh
    total_images = 0
    for ext in exts:
        images = list(image_dir.rglob(f"*{ext}"))
        total_images += len(images)
    
    # Hiển thị kết quả
    print(f"Tổng số ảnh trong {IMAGE_DIR}: {total_images}")

# Gọi hàm
if __name__ == "__main__":
    count_images()