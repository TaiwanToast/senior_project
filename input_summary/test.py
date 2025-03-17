import os
import json
import openai
import re

api_key = ""
client = openai.OpenAI(api_key=api_key)

# 需求資料
key_required = ["品牌", "使用場合", "預算範圍"]
basic_data = {"品牌": "未知", "使用場合": "未知", "預算範圍": "未知"}
extra_data = []

# 訊息歷史
messages = [
    {"role": "system", "content": 
    """
    你是一個使用繁體中文的 AI 助理，專門解析使用者的購物需求。
    你的目標是從使用者輸入中提取至少 **2 項** 基本資訊，然後提前結束對話，並輸出結果。
    如果不足 2 項，請進行最多 **3 次追問**，如果仍然不足 2 項，就直接輸出當前已知資訊。

    ### **解析需求時，請提取以下基本資訊：**
    - **1.品牌**
    - **2.使用場合**
    - **3.預算範圍**
    - **4.顏色**
    若使用者提出其他需求，也請一併紀錄，但不算在基本資訊中。

    #### **提前結束條件：**
    1. 如果使用者 **已提供 2 項以上** 的基本資訊，請直接結束對話並輸出解析結果。
    2. 如果使用者的輸入不足 2 項，請進行追問，最多 **3 次**。
    3. 如果追問 **3 次後仍不足 2 項**，請直接輸出當前已知資訊。

    #### **特殊處理規則：**
    - 若使用者詢問與 **商品查詢無關** 的內容，請回應 **「抱歉，無法提供此功能」**，不要進行需求解析。
    
    #### **輸出格式：**
    - 資訊欄位：需求
    例如(
    品牌：不拘
    價格：3000
    用途：可以打羽球
    其他用途：且需要膠底的球鞋)
    
    """
    }
]

def update_basic_data(output):
    """根據 AI 統整的需求更新 basic_data"""
    patterns = {
        "品牌": r"品牌：(.+?)(?:\n|$)",
        "使用場合": r"使用場合：(.+?)(?:\n|$)",
        "預算範圍": r"預算範圍：(.+?)(?:\n|$)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            basic_data[key] = match.group(1).strip()

def save_data(output):
    data = {"basic_data": {}, "extra_data": []}

    pattern = r"(.+?)：(.+)"
    matches = re.findall(pattern, output)

    for key, value in matches:
        key = key.strip()
        value = value.strip()

        # 若欄位屬於基本資訊，存入 basic_data
        if key in ["品牌", "價格", "顏色", "用途"]:
            data["basic_data"][key] = value
        else:  # 其他需求存入 extra_data
            data["extra_data"].append(f"{key}：{value}")

    # 儲存為 JSON 檔案
    with open("shopping_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("需求已儲存至 shopping_data.json")


def response(user_input):
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
    )

    output = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": output})

    if "？" not in output:
        save_data(output)

    return output


def main():
    """主程式執行流程"""
    print("AI : 你好！請描述你的購物需求")

    for _ in range(3):
        user_input = input("使用者 : ")
        output = response(user_input)

        print(f"AI : {output}")

        if "？" not in output:
            return


if __name__ == "__main__":
    main()  # 我想買一雙球鞋
