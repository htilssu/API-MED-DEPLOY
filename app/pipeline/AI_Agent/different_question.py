import os
import random
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------- CẤU HÌNH ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_discriminative_questions(
    caption: str,
    labels: list[str],
    group_disease_name: str,
    model=None
) -> list[str]:
    if model is None:
        model = genai.GenerativeModel("gemini-2.5-pro")

    variation_prompts = [
        "Hãy đặt câu hỏi giúp loại trừ ngay lập tức một bệnh nếu câu trả lời phù hợp.",
        "Tập trung vào triệu chứng hoặc dấu hiệu chỉ có ở một bệnh duy nhất trong nhóm này.",
        "Đặt câu hỏi về đặc điểm đặc trưng (pathognomonic) mà chỉ một bệnh có.",
        "Hãy hỏi về dấu hiệu 'red flag' giúp xác định hoặc loại trừ bệnh ngay.",
        "Đặt câu hỏi mà nếu trả lời 'có' hoặc 'không' sẽ giúp chẩn đoán chính xác nhất.",
        "Tập trung vào các yếu tố quyết định, không hỏi chung chung.",
        "Hãy hỏi về triệu chứng hoặc yếu tố mà chỉ một bệnh trong danh sách này có.",
        "Đặt câu hỏi giúp phân biệt nhanh nhất, tránh các triệu chứng trùng lặp.",
        "Hãy hỏi như một bác sĩ muốn xác định bệnh ngay lập tức chỉ qua một vài câu hỏi then chốt.",
        "Hãy đặt câu hỏi như khi đang phỏng vấn bệnh sử để phân biệt bệnh rõ ràng.",
        "Tập trung vào các yếu tố loại trừ, ví dụ: vị trí tổn thương, cảm giác, diễn tiến theo thời gian.",
        "Đặt câu hỏi theo hướng lâm sàng thực tế: nếu đặc điểm này có thì bệnh nào sẽ bị loại bỏ?",
        "Hãy hỏi như một bác sĩ giàu kinh nghiệm đang phân vân giữa các chẩn đoán.",
        "Đặt câu hỏi nhằm làm rõ khác biệt điển hình giữa các bệnh trên.",
        "Tập trung vào dấu hiệu mang tính đặc hiệu giúp phân biệt nhanh.",
        "Hãy đưa ra câu hỏi gợi mở triệu chứng ít thấy nhưng quan trọng với nhóm bệnh {group_disease_name}.",
        "Hỏi sâu về yếu tố khởi phát, tính chất lan rộng, yếu tố môi trường đi kèm.",
        "Hãy hỏi các biểu hiện mà chỉ 1 trong các bệnh này có khả năng gặp.",
    ]
    variation = random.choice(variation_prompts)
    print(f"Variation prompt: {variation}")

    prompt = f"""
Bạn là bác sĩ da liễu chuyên về nhóm bệnh **{group_disease_name}**.
Bạn đang xem xét một ảnh da liễu với mô tả như sau:

--- MÔ TẢ ẢNH ---
{caption}

Tôi đang phân vân giữa các bệnh sau: {', '.join(labels)}.

{variation}

Yêu cầu:
- Viết ra 3 câu hỏi nhằm **phân biệt rõ ràng** các bệnh trên.
- Ưu tiên các câu hỏi giúp xác định hoặc loại trừ bệnh ngay lập tức, tránh hỏi chung chung.
- Tập trung vào triệu chứng, dấu hiệu đặc trưng, hoặc yếu tố chỉ có ở một bệnh.
- Mỗi câu hỏi cần hướng đến triệu chứng, cảm giác, vị trí, thời điểm, diễn tiến... có khả năng **loại trừ bệnh này so với bệnh khác**.
- Không được hỏi câu chung chung như “bạn có ngứa không?” — thay vào đó hãy hỏi theo cách **giúp nhận diện đặc trưng riêng**.
- Câu hỏi phải: rõ ràng, không lặp ý, không mô tả lại ảnh, không giải thích.
- **Chỉ dùng từ “bạn” để hỏi.** Tránh mọi từ như “bệnh nhân”, “em bé”, “anh/chị”, v.v.
- Viết bằng **tiếng Việt**, đúng 3 câu hỏi, không có mô tả thừa.

Chỉ ghi ra 3 câu hỏi:
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        questions = [
            q.strip().lstrip("-•1234567890. ") for q in raw_text.split("\n") if q.strip()
        ]
        return questions[:3]
    except Exception as e:
        logging.error(f"Lỗi sinh câu hỏi phân biệt: {e}")
        return []
