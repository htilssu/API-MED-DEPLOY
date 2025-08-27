from io import BytesIO
import google.generativeai as genai
from PIL import Image
from typing import Optional
import logging
from app.config.setting import setting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=setting.GEMINI_API_KEY)

def generate_description_with_Gemini(image_data: bytes):
    try:
        img = Image.open(BytesIO(image_data))
        model = genai.GenerativeModel('gemini-2.5-pro')
        prompt = """
Hãy kiêm tra bức ảnh nếu là ảnh da liễu thì tiếp tục mô tả còn nếu không phải thì trả về "Không phải ảnh da liễu".
Hãy quan sát kỹ bức ảnh da bên dưới và mô tả lại một cách trung lập, chính xác, chỉ dựa trên những gì có thể nhìn thấy bằng mắt thường trong ảnh, bằng tiếng Việt.

Yêu cầu mô tả:
- Vị trí tổn thương (xuất hiện ở vùng nào trên cơ thể).
- Số lượng và kích thước tổn thương (một hay nhiều, ước lượng kích thước).
- Màu sắc chủ đạo và sự đồng nhất màu sắc.
- Đặc điểm bề mặt da (trơn láng, khô, bong vảy, sần sùi, đóng mày, loét, mụn nước, mủ...).
- Bờ tổn thương (rõ hay mờ, đều hay không đều, dạng vòng hay lan tỏa).
- Tính đối xứng (có xuất hiện hai bên cơ thể không).
- Kiểu phân bố (rải rác, tập trung thành cụm, theo mảng lớn, theo đường...).
- Dấu hiệu bất thường khác (sưng nề, chảy dịch, mủ vàng, hoại tử, lở loét, đường hầm dưới da...).

Lưu ý cực kỳ quan trọng:
- Chỉ mô tả những gì quan sát được bằng mắt thường trong ảnh.
- Không được đưa ra bất kỳ chẩn đoán, suy luận y khoa, hoặc gợi ý bệnh lý nào.
- Không sử dụng kinh nghiệm y khoa, không dự đoán nguyên nhân.
- Không sử dụng bullet point, markdown, ký hiệu đặc biệt, hoặc xuống dòng.
- Trả về kết quả dưới dạng một đoạn văn y khoa mô tả, rõ ràng, trung lập, duy nhất.

Chỉ trả về đoạn mô tả, không thêm bất kỳ thông tin nào khác.
"""
        response = model.generate_content([prompt, img])
        caption = response.text.replace("\n", " ").strip()
        return caption
    except Exception as e:
        logging.error(f"Lỗi khi tạo caption với Gemini: {e}")
        return None