import google.generativeai as genai
from PIL import Image
from typing import Optional
import logging
import os
from dotenv import load_dotenv
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def extract_label_name(label):
    if isinstance(label, tuple) and isinstance(label[0], (str, np.str_)):
        return str(label[0])
    elif isinstance(label, str):
        return label
    return str(label)  
def generate_diagnosis_with_gemini(description, sorted_labels):
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')  
        labels_only = [extract_label_name(label) for label in sorted_labels[:30]]
        labels_text = "\n".join([f"- {label}" for label in labels_only])
        print(f"label text: {labels_text} \n description: {description}")
        prompt = f"""
Bạn là một chuyên gia da liễu.

Dưới đây là mô tả tổn thương da từ ảnh đầu vào:
\"\"\"{description}\"\"\"

Hệ thống AI đã dự đoán một số bệnh có thể gặp (từ ảnh toàn phần và vùng tổn thương), cùng độ phù hợp:
{labels_text}

---
## Nhiệm vụ của bạn:
1. Đọc kỹ mô tả tổn thương và danh sách nhãn dự đoán.
2. Phân tích từng đặc điểm lâm sàng, so sánh với các tiêu chí của 4 nhóm bệnh chính bên dưới.
3. Áp dụng các luật loại trừ để loại bỏ các nhóm không phù hợp.
4. Đưa ra lý do ngắn gọn cho việc chọn nhóm bệnh cuối cùng (chỉ để kiểm tra, KHÔNG đưa vào câu trả lời cuối cùng).
5. Chỉ trả về đúng một dòng duy nhất là tên nhóm bệnh phù hợp nhất.

---
### 4 nhóm bệnh chính:
1. **fungal-infections (nhiễm nấm):**
   - Da khô, bong vảy mịn như phấn, bờ rõ/hình vòng, trung tâm lành hơn ngoại vi, không mủ, không sưng nóng đỏ, ngứa nhẹ/vừa, vị trí: da đầu, thân mình, chi, bẹn.
2. **virus (nhiễm virus):**
   - Mụn nước, bóng nước, sẩn/loét nông, đau rát/ngứa, phân bố dọc dây thần kinh/đối xứng, không mủ vàng/vảy tiết dày.
3. **bacterial-infections (nhiễm vi khuẩn):**
   - Da sưng nóng đỏ đau, có mủ, đóng mày vàng, hoại tử nhẹ, đơn độc (đặc biệt ở da đầu), vảy dày trắng vàng, bề mặt sần sùi, bờ rõ/không đều, lan rộng nhanh, không mủ nhưng có màu trắng vàng vẫn cân nhắc nhóm này.
4. **parasitic-infections (nhiễm ký sinh trùng):**
   - Ngứa nhiều (đặc biệt ban đêm), sẩn nhỏ/rãnh/vết xước do gãi, vị trí: kẽ ngón tay, bẹn, quanh rốn, mông, lây lan nhanh qua tiếp xúc.

---
### Luật loại trừ:
- Không có mủ, sưng, đau, dịch vàng → loại trừ bacterial-infections
- Không có ngứa dữ dội, đường hầm → loại trừ parasitic-infections
- Không có ban đỏ dạng mụn nước, không phân bố đối xứng/dải → loại trừ virus
- Tổn thương có vảy, bờ rõ, không có dịch → nghiêng về fungal-infections

---
### Lưu ý đặc biệt:
- Nếu tổn thương ở da đầu, mặt, cổ, nhỏ/đơn lẻ/rải rác, có vảy, không sưng/mủ rõ, gần nang lông/chân tóc → ưu tiên bacterial-infections (viêm nang lông).
- Nếu tổn thương nhỏ (<5mm), hình tròn, rải rác, không vảy, không mủ, không sưng, không đối xứng → cân nhắc parasitic-infections (ghẻ/côn trùng cắn).
- KHÔNG nhầm sang fungal nếu KHÔNG có vảy rõ, ranh giới tăng sừng, hoặc mảng lớn lan tỏa.
- KHÔNG nhầm sang virus nếu KHÔNG có cụm sẩn, ban, mụn nước, hoặc tổn thương đa hình thái.

---
### QUAN TRỌNG:
- Chỉ trả về đúng **một dòng duy nhất** là một trong các nhóm sau:
  - fungal_infections
  - virus
  - bacterial_infections
  - parasitic_infections
- ❌ Không viết hoa, không thêm dấu câu, không giải thích, không dùng từ gần nghĩa.
- ⚠️ Nếu hiểu rõ, hãy suy luận từng bước, sau đó chỉ trả về tên nhóm bệnh phù hợp nhất.

---
### Ví dụ trả lời:
fungal_infections
"""       
        response = model.generate_content(prompt)
        caption = response.text.replace("\n", " ").strip().lower()
        return caption

    except Exception as e:
        logging.error(f"❌ Lỗi khi tạo chẩn đoán với Gemini: {e}")
        return "unknown"