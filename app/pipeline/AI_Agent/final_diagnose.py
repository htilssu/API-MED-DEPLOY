import google.generativeai as genai
import logging
from app.config.setting import setting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=setting.GEMINI_API_KEY)

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
**CHUYÊN MÔN: NHIỄM NẤM DA**

---

**1. Tinea Pedis (Nấm kẽ chân - Athlete's Foot):**
- **Khởi đầu:** Người bệnh thường thấy hơi ngứa nhẹ giữa các ngón chân, nhất là ngón 3 và 4. Cảm giác nóng nhẹ, ẩm ướt, khó chịu khi đi giày lâu.
- **Tiến triển:** Da bắt đầu bong vảy mịn, có thể nứt da, chảy dịch nhẹ, có mùi hôi. Nếu không điều trị, có thể lan ra mu bàn chân hoặc các ngón khác.
- **Cảm giác:** Ngứa tăng khi đi tất giày, đặc biệt thời tiết nóng. Một số người đau nhẹ do nứt kẽ.
- **Dấu hiệu chính:** Vùng tổn thương bong da trắng, mềm, ẩm. Bờ ranh giới rõ.

---

**2. Tinea Corporis (Nấm thân - Ringworm):**
- **Khởi đầu:** Xuất hiện một mảng đỏ tròn nhỏ trên da, thường không đau, chỉ hơi ngứa nhẹ.
- **Tiến triển:** Mảng lớn dần, bờ tổn thương nổi gờ lên, trung tâm mờ nhạt hơn – tạo hình vòng tròn. Vảy khô quanh rìa, có thể có nhiều mảng đồng thời.
- **Cảm giác:** Ngứa nhẹ đến vừa, khó chịu khi mồ hôi nhiều hoặc ma sát.
- **Dấu hiệu chính:** Mảng tròn, rìa đỏ, có vảy, trung tâm lành hơn rìa. Có thể lan hoặc tái phát.

---

**3. Tinea Capitis (Nấm da đầu):**
- **Khởi đầu:** Xuất hiện vùng rụng tóc giới hạn, có thể ngứa nhẹ hoặc gàu nhiều.
- **Tiến triển:** Tóc gãy sát da đầu, vùng tổn thương mở rộng, có vảy trắng hoặc chấm đen, đôi khi nổi mủ (kerion).
- **Cảm giác:** Ngứa, hơi đau nếu có viêm. Trẻ nhỏ hay gãi đầu, có thể nổi hạch cổ.
- **Dấu hiệu chính:** Rụng tóc dạng mảng, có vảy, đôi khi mủ, để lại sẹo nếu không điều trị.

---

**4. Onychomycosis (Nấm móng):**
- **Khởi đầu:** Móng đổi màu vàng nhạt, dày nhẹ ở đầu móng.
- **Tiến triển:** Móng ngày càng dày, giòn, dễ gãy. Có thể tách ra khỏi nền móng, gây biến dạng móng.
- **Cảm giác:** Thường không đau, nhưng có thể cộm, khó chịu khi cắt móng hoặc mang giày.
- **Dấu hiệu chính:** Móng dày, giòn, màu vàng nâu/xám, có thể tách ra khỏi nền móng.

---

**5. Candidiasis (Nhiễm Candida ở da):**
- **Khởi đầu:** Mảng đỏ ở vùng nếp gấp da (bẹn, nách, dưới ngực), hơi ngứa, ẩm.
- **Tiến triển:** Mảng đỏ lan rộng, bề mặt bóng, ướt, có viền vảy trắng mịn xung quanh, có thể nổi mụn mủ nhỏ quanh rìa.
- **Cảm giác:** Ngứa nhiều, rát, cảm giác bỏng da khi chạm nước hoặc ma sát.
- **Dấu hiệu chính:** Tổn thương đỏ, bóng, ẩm, có viền ranh giới và tổn thương vệ tinh xung quanh.
"""
    elif group_disease_name == "bacterial_infections":
        specialized_context = """
**CHUYÊN MÔN: NHIỄM KHUẨN DA**

---

**1. Impetigo (Chốc lở):**
- **Khởi đầu:** Bắt đầu bằng một vết đỏ hoặc mụn nước nông, thường quanh miệng, mũi.
- **Tiến triển:** Mụn nước vỡ nhanh tạo vảy vàng như mật ong. Có thể lan sang vùng da lân cận nếu gãi.
- **Cảm giác:** Ngứa nhiều, trẻ nhỏ hay gãi. Thường không đau.
- **Dấu hiệu chính:** Vảy màu vàng mật ong, nền đỏ, tổn thương nông, dễ lây lan.

---

**2. Cellulitis (Viêm mô tế bào):**
- **Khởi đầu:** Đột ngột xuất hiện vùng da đỏ, sưng, nóng, thường ở chân.
- **Tiến triển:** Vùng đỏ lan rộng nhanh, đau tăng. Có thể kèm sốt, mệt mỏi, nổi hạch khu vực.
- **Cảm giác:** Đau nhức, nóng rát, cảm giác da căng.
- **Dấu hiệu chính:** Vùng da đỏ, sưng, bờ không rõ, sờ nóng, đau. Da có thể bóng lên.

---



**3. Folliculitis (Viêm nang lông):**
- **Khởi đầu:** Mụn nhỏ đỏ hoặc mụn mủ ở gốc lông, đơn lẻ hoặc rải rác.
- **Tiến triển:** Một số mụn vỡ, đóng vảy. Nếu lan sâu → nhọt (furuncle).
- **Cảm giác:** Ngứa hoặc đau nhẹ tại vị trí mụn.
- **Dấu hiệu chính:** Mụn mủ nhỏ ở nang lông, đỏ quanh gốc lông, phân bố ở mông, đùi, nách.

---


"""
    elif group_disease_name == "virus":
        specialized_context = """
**CHUYÊN MÔN: NHIỄM VIRUS DA**

---

**1. Herpes Simplex (Mụn rộp môi/sinh dục):**
- **Khởi đầu:** Cảm giác ngứa rát, châm chích ở vùng môi/sinh dục 1–2 ngày trước khi tổn thương xuất hiện.
- **Tiến triển:** Xuất hiện mụn nước nhỏ, nhóm lại, dễ vỡ → loét nông → đóng mày.
- **Cảm giác:** Rát bỏng, nhức. Đau nhẹ tại vị trí tổn thương.
- **Dấu hiệu chính:** Mụn nước nhỏ nhóm lại trên nền đỏ, vỡ nhanh thành vết loét nông, tái phát cùng vị trí.

---

**2. Herpes Zoster (Zona):**
- **Khởi đầu:** Đau rát, tê châm chích vùng da theo dây thần kinh → sau 2–3 ngày nổi mụn nước.
- **Tiến triển:** Mụn nước thành cụm, dọc theo 1 bên cơ thể. Sau vài ngày mụn hóa mủ → vỡ → đóng vảy.
- **Cảm giác:** Đau rát dữ dội, có thể kéo dài sau khi lành (đau sau zona).
- **Dấu hiệu chính:** Mụn nước theo dải da, một bên cơ thể, thường ở ngực, lưng, trán.

---

**3. Chickenpox (Thủy đậu):**
- **Khởi đầu:** Sốt, mệt mỏi trước khi xuất hiện ban đỏ → mụn nước.
- **Tiến triển:** Mụn nước xuất hiện toàn thân, các tổn thương ở nhiều giai đoạn (ban, nước, mủ, vảy).
- **Cảm giác:** Ngứa dữ dội, khó chịu. Trẻ nhỏ hay quấy khóc.
- **Dấu hiệu chính:** Mụn nước rải rác toàn thân, không đối xứng, nhiều giai đoạn tiến triển.

---

**4. Monkeypox (Đậu mùa khỉ):**
- **Khởi đầu:** Sốt cao, đau cơ, nổi hạch → 1–3 ngày sau xuất hiện tổn thương da.
- **Tiến triển:** Mụn nước, sau đó mụn mủ, loét, dày vảy. Rải rác hoặc tập trung mặt, tay, bộ phận sinh dục.
- **Cảm giác:** Đau, sưng hạch, mệt mỏi toàn thân.
- **Dấu hiệu chính:** Mụn nước – mủ chắc, loét, tổn thương rõ, thường đối xứng, kèm hạch to.

---

**5. Sarampion (Sởi):**
- **Khởi đầu:** Sốt, ho, sổ mũi, viêm kết mạc trước khi nổi ban 3–4 ngày.
- **Tiến triển:** Ban dát sẩn xuất hiện từ mặt → thân mình → tay chân, ngứa nhẹ, kéo dài 5–7 ngày.
- **Cảm giác:** Khó chịu toàn thân, ho khan, đỏ mắt.
- **Dấu hiệu chính:** Ban đỏ rải rác, xuất hiện theo thứ tự đầu → chân, kèm triệu chứng hô hấp – kết mạc.

---

**6. Warts (Mụn cóc):**
- **Khởi đầu:** Giai đoạn tiền triệu kéo dài 3–4 ngày với sốt cao, ho khan, chảy nước mũi, viêm kết mạc mắt (đỏ mắt, chảy nước mắt), xuất hiện hạt Koplik ở niêm mạc miệng (dấu hiệu đặc trưng).
- **Tiến triển:** Sau giai đoạn tiền triệu, ban dát sẩn đỏ xuất hiện sau tai và trán, lan dần xuống mặt, cổ, thân mình rồi tới tay chân theo thứ tự từ trên xuống dưới. Ban kéo dài 5–7 ngày, có thể bong vảy khi lui.
- **Cảm giác:** Ban dát sẩn đỏ, mọc theo trình tự đầu → thân → chi, kèm sốt cao, viêm kết mạc, hạt Koplik và triệu chứng hô hấp.

"""
    elif group_disease_name == "parasitic_infections":
        specialized_context = """
**CHUYÊN MÔN: NHIỄM KÝ SINH TRÙNG DA**

---


**1. Head Lice (Chấy/rận):**
- **Khởi đầu:** Ngứa da đầu hoặc thân mình, âm ỉ, đặc biệt vào buổi tối.
- **Tiến triển:** Tổn thương do gãi: vết xước, sẩn đỏ, tróc da. Có thể thấy trứng chấy bám chắc ở chân tóc.
- **Cảm giác:** Ngứa nhiều, râm ran như có vật bò.
- **Dấu hiệu chính:** Trứng chấy bám gần da đầu, vết xước vùng gáy, sau tai.

---

**2. Infestations/Bites (Côn trùng cắn):**
- **Khởi đầu:** Nốt đỏ sưng tại vị trí cắn, nổi gồ trên bề mặt da.
- **Tiến triển:** Có thể tạo trung tâm mụn nước hoặc mủ nhỏ. Một số người phản ứng mạnh → phù nề.
- **Cảm giác:** Ngứa, rát, đôi khi đau buốt.
- **Dấu hiệu chính:** Nốt đỏ sưng trung tâm, có thể thành mụn nước/mủ. Thường phân bố ở vùng hở (tay, chân).

- **Loại trừ chung:** Nếu không có ngứa hoặc không có phân bố điển hình → ít nghi ngờ ký sinh trùng.
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
**Chỉ trả lời tên bệnh chính xác duy nhất (không giải thích thêm, không đề xuất bệnh ngoài nhóm này). Kết quả bệnh là tiếng anh**
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip().split("\n")[0]
    except Exception as e:
        logging.error(f"Lỗi chọn nhãn cuối bằng Gemini: {e}")
        return ""
