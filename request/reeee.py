import openai
import pdfplumber
import re

api_key = ""
client = openai.OpenAI(api_key=api_key,base_url="")

# 擷取商品資訊
def extract_data_from_pdf(pdf_path):
    data = {}
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        # 商品名稱
        match = re.search(r"商品名稱\s*(.+?)\s*店家名稱", full_text)
        if match:
            data["商品名稱"] = match.group(1).strip()

        # 商品售價
        match = re.search(r"商品售價\s*([\d,]+)\s*店家電話", full_text)
        if match:
            data["商品售價"] = match.group(1).strip()

        # 商品描述：從商品資料表結束欄位起始擷取，排除標準欄位
        split_pattern = r"商品資料表"
        split_text = re.split(split_pattern, full_text)
        if len(split_text) > 1:
            product_description = split_text[1].strip()
            # 移除可能的干擾欄位名稱
            product_description = re.sub(r"(商品名稱|店家名稱|商品售價|店家電話|店家地址|商品描述)", "", product_description)
            data["商品描述"] = product_description.strip()

    return data

# 生成推薦理由
def generate_comparison_recommendation(product_info, user_input):
    text = f"""
        使用者需求：{user_input}\n
        請你大量採用主觀意見，
        根據使用者需求以及以下3個商品資訊，
        完整提供3個商品的優缺點，
        並以3個商品之間的比較給出推薦，
        在推薦理由要說明為什麼不選另外兩樣商品：\n
        """
    
    for i, info in enumerate(product_info, 1):
        text += f"\n【商品 {i}】\n"
        text += f"商品名稱：{info.get('商品名稱', '未知')}\n"
        text += f"售價：{info.get('商品售價', '未知')} 元\n"
        text += f"商品描述：{info.get('商品描述', '無')}\n"

    messages = [{"role": "system", "content": text}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def main():
    # pdf_path_1 = "羅技 Logitech H340 USB耳機麥克風.pdf"
    # pdf_path_2 = "Razer 雷蛇 BlackShark V2 X 黑鯊 電競耳機 3.5mm.pdf"
    # pdf_path_3 = "SADES DIABLO 暗黑鬥狼RGB REALTEK 電競耳麥 7.1 (USB) SA-916.pdf"
    # product_info = extract_data_from_pdf(pdf_path_1)
    # product_info = extract_data_from_pdf(pdf_path_2)
    # product_info = extract_data_from_pdf(pdf_path_3)

    paths = [
        "羅技 Logitech H340 USB耳機麥克風.pdf",
        "Razer 雷蛇 BlackShark V2 X 黑鯊 電競耳機 3.5mm.pdf",
        "SADES DIABLO 暗黑鬥狼RGB REALTEK 電競耳麥 7.1 (USB) SA-916.pdf"
    ]

    product_info = [extract_data_from_pdf(path) for path in paths]
    # print(product_info[2]["商品描述"])

    user_input = "耳罩式耳機，品牌：不拘，價格：2000以內，特殊需求：附帶麥克風"
    # print(product_info)

    result = generate_comparison_recommendation(product_info, user_input)
    print(result)

if __name__ == "__main__":
    main()