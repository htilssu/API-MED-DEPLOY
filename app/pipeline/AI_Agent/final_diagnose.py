import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def select_final_diagnosis_with_llm(
    caption: str,
    labels: list[str],
    questions: list[str],
    answers: list[str],
    group_disease_name: str,
) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")

    qa_text = "\n".join(
        [f"- {q}\n  → {a}" for q, a in zip(questions, answers)]
    )
    if group_disease_name == "fungal_infections":
        specialized_context = """
**CHUYÊN MÔN NHIỄM NẤM:**
- Tập trung vào đặc điểm: bờ rõ, hình tròn/vòng, bong vảy, vị trí ẩm
- Các bệnh điển hình: Athlete's Foot, Ringworm, Nail Fungus, Tinea Capitis, Candidiasis
- Loại trừ nếu có mủ, sưng nóng đỏ đau, hoặc tổn thương sâu
- Ưu tiên bệnh có vảy khô, ranh giới rõ, không có dịch mủ
"""
    elif group_disease_name == "bacterial_infections":
        specialized_context = """
**CHUYÊN MÔN NHIỄM VI KHUẨN:**
- Tập trung vào đặc điểm: sưng nóng đỏ đau, có mủ, vảy vàng, tổn thương sâu
- Các bệnh điển hình: Cellulitis, Impetigo, Folliculitis
- Ưu tiên bệnh có mủ, loét, hoại tử, hoặc tổn thương lan rộng nhanh
- Loại trừ nếu chỉ có vảy khô, không sưng, không mủ
"""
    elif group_disease_name == "virus":
        specialized_context = """
**CHUYÊN MÔN NHIỄM VIRUS:**
- Tập trung vào đặc điểm: mụn nước, ban đỏ, phân bố đối xứng hoặc theo dây thần kinh
- Các bệnh điển hình: Herpes, Herpes Zoster, Chickenpox, Monkeypox, Sarampion
- Ưu tiên bệnh có mụn nước, ban đỏ, tổn thương đa hình thái
- Loại trừ nếu có mủ vàng, vảy dày, hoặc tổn thương đơn lẻ không đối xứng
"""
    elif group_disease_name == "parasitic_infections":
        specialized_context = """
**CHUYÊN MÔN NHIỄM KÝ SINH TRÙNG:**
- Tập trung vào đặc điểm: ngứa dữ dội, đường hầm, sẩn nhỏ, vết xước
- Các bệnh điển hình: Head Lice, Infestations_Bites, Scabies
- Ưu tiên bệnh có ngứa nhiều, đường hầm dưới da, tổn thương rải rác
- Loại trừ nếu không ngứa, không có đường hầm, hoặc tổn thương lớn lan tỏa
"""
    else:
        specialized_context = ""

    prompt = f"""
Bạn là bác sĩ da liễu chuyên về nhóm bệnh **{group_disease_name}**. 
Chỉ sử dụng kiến thức chuyên sâu về nhóm bệnh này để đưa ra chẩn đoán. 
Không được đề xuất các bệnh ngoài nhóm **{group_disease_name}**.

{specialized_context}

Dưới đây là mô tả ảnh tổn thương da, danh sách bệnh nghi ngờ và các thông tin phân biệt thu được từ người bệnh.

--- MÔ TẢ ẢNH ---
{caption}

--- CÁC BỆNH NGHI NGỜ ---
{', '.join(labels)}

--- CÂU TRẢ LỜI PHÂN BIỆT ---
{qa_text}

Dựa vào tất cả thông tin trên, với vai trò là chuyên gia về nhóm bệnh **{group_disease_name}**, hãy chọn ra bệnh hợp lý nhất từ danh sách bệnh nghi ngờ.  
**Chỉ trả lời tên bệnh chính xác duy nhất (không giải thích thêm, không đề xuất bệnh ngoài nhóm này).**
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip().split("\n")[0]
    except Exception as e:
        logging.error(f"Lỗi chọn nhãn cuối bằng Gemini: {e}")
        return ""
