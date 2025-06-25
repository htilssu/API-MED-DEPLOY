import requests

def get_uv_index(lat: float, lon: float, api_key: str):
    url = "https://api.openweathermap.org/data/2.5/uvi"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        uv_value = data.get("value")
        print(f"Vị trí: ({lat}, {lon})")
        print(f"Chỉ số UV: {uv_value}")
        return uv_value
    else:
        print(f"Lỗi khi gọi API: {response.status_code}")
        print(response.text)
        return None

def uv_warning_level(uv_value: float) -> str:
    """
    Trả về thông báo cảnh báo dựa trên giá trị chỉ số UV.
    """
    if uv_value is None:
        return "⚠️ Không thể xác định chỉ số UV."

    if uv_value <= 2:
        return "🟢 Mức UV thấp. Bạn có thể ở ngoài trời an toàn."
    elif uv_value <= 5:
        return "🟡 Mức UV trung bình. Hãy đội nón và dùng kem chống nắng."
    elif uv_value <= 7:
        return "🟠 Mức UV cao. Tìm bóng râm, hạn chế ra nắng từ 10h–16h."
    elif uv_value <= 10:
        return "🔴 Mức UV rất cao. Nên ở trong nhà và bảo vệ da kỹ lưỡng."
    else:
        return "🟣 Mức UV cực kỳ nguy hiểm! Tránh ra ngoài và mặc kín toàn thân."
    
def check_and_warn_uv(lat: float, lon: float, api_key: str):
    uv_value = get_uv_index(lat, lon, api_key)
    warning_message = uv_warning_level(uv_value)
    print(warning_message)
    return warning_message