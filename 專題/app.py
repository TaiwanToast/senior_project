from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import openai
import json
import os
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# ---------------------------
# 請填入你的 OpenAI API 金鑰
api_key = "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"
client = openai.OpenAI(api_key=api_key)

# ---------------------------
# 初始系統訊息
messages = [
{"role": "system", "content": """
你是購物推薦助理，目標是推薦最合適的產品，但必須先蒐集完整需求。

請遵循以下規則進行對話：
1. 詢問至少 3 個問題，包含：
   - 使用用途（打遊戲？辦公？影音剪輯？）
   - 預算範圍
   - 品牌偏好
2. 每次對話僅問一個問題
3. 收集滿 3 項後，再一次性推薦 2~4 項商品
4. 推薦每項商品都需包含以下資訊：
    - 商品名稱（精簡且真實）
    - 價格（新台幣，數字格式）
    - 推薦語（簡短有說服力）
    - 品圖片網址（請務必來自「PChome、Momo、蝦皮」等真實電商平台）
    - 連結（請加上正確可點擊的購物連結網址，商品連結網址，PChome、Momo、蝦皮等）

5. 回傳格式範例如下，每項商品之間請用 `---` 分隔：
---
商品：
名稱：MSI GeForce RTX 4070
價格：18990
推薦語：效能與散熱兼具，適合 2K/4K 高畫質遊戲需求
圖片：https://img.pchome.com.tw/cs/items/DSAURAA900I1L56/000001_1730712514.jpg
連結：https://24h.pchome.com.tw/prod/DSAURA-A900I1L56
---

如果使用者需求與購物無關，請委婉回應不提供相關服務。
"""}
]

# 全域變數：目前分類（預設 None）、收藏清單（儲存產品字典）
current_category = None
collection = []

# JSON 收藏檔案路徑
COLLECTION_FILE = "collection.json"

# ---------------------------
@app.route("/")
def index():
    return render_template("index.html", user=session.get("user"))

@app.route("/chat", methods=["POST"])
def chat():
    global current_category
    user_input = request.form.get("user_input", "")
    user_bubble = f'<div class="bg-gray-700 p-3 rounded self-end max-w-lg text-white">{user_input}</div>'

    if current_category:
        user_input += f" 類別：{current_category}"

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )

    gpt_reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": gpt_reply})

    if "---" in gpt_reply:
        keyword = extract_keyword_from_reply(gpt_reply)
        real_products = (search_pchome(keyword) or []) + (search_momo(keyword) or [])
        return user_bubble + gpt_to_html_cards(gpt_reply) + generate_real_cards(real_products) + scroll_to_bottom_script()
    else:
        return user_bubble + f'<div class="bg-gray-600 p-3 rounded self-start max-w-lg text-white">{gpt_reply}</div>' + scroll_to_bottom_script()

def extract_keyword_from_reply(text):
    lines = text.splitlines()
    for line in lines:
        if "名稱：" in line:
            return line.split("：", 1)[1].strip()
    return "顯卡"


def extract_keyword_from_reply(text):
    lines = text.splitlines()
    for line in lines:
        if "名稱：" in line:
            return line.split("：", 1)[1].strip()
    return "顯卡"

def gpt_to_html_cards(text):
    html = '<div class="bg-gray-700 p-4 rounded space-y-4">'
    products = text.split("---")
    for item in products:
        if not item.strip():
            continue
        name = price = desc = img = link = ""
        for line in item.strip().splitlines():
            if "名稱：" in line:
                name = line.split("：", 1)[1].strip()
            elif "價格：" in line:
                price = line.split("：", 1)[1].replace("$", "").strip()
            elif "推薦語：" in line:
                desc = line.split("：", 1)[1].strip()
            elif "圖片：" in line:
                img = line.split("：", 1)[1].strip()
            elif "連結：" in line:
                link = line.split("：", 1)[1].strip()

        # 若沒填連結就不顯示
        if not name or not price or not desc or not img or not link:
            continue

        html += f"""
        <div class="bg-gray-800 p-3 rounded flex flex-col gap-2">
          <a href="{link}" target="_blank" class="flex items-center gap-4">
            <img src="{img}" onerror="this.src='https://via.placeholder.com/100'" class="w-24 h-24 object-contain bg-white rounded">
            <div>
              <div class="text-lg font-bold text-white">{name}</div>
              <div class="text-green-400">${price}</div>
              <div class="text-sm text-gray-300">{desc}</div>
            </div>
          </a>
          <form hx-post="/collect" hx-target="#collection_list" hx-swap="innerHTML">
            <input type="hidden" name="name" value="{name}">
            <input type="hidden" name="price" value="{price}">
            <input type="hidden" name="desc" value="{desc}">
            <input type="hidden" name="img" value="{img}">
            <button type="submit" class="bg-blue-500 px-2 py-1 text-xs rounded">加入收藏</button>
          </form>
        </div>
        """

    html += '</div>'
    return html


def generate_real_cards(products):
    html = '<div class="bg-gray-900 p-4 rounded space-y-4">'
    html += '<div class="text-white font-semibold mb-2">🔍 更多商品推薦</div>'
    for item in products:
        if not any(x in item['img'] for x in ['pchome', 'momoshop']):
            continue

        # 如果原價比特價高，顯示折扣
        price_html = f"<div class='text-green-400'>${item['price']}</div>"
        if str(item["origin_price"]).isdigit() and int(item["origin_price"]) > int(item["price"]):
            price_html = f"""
                <div>
                    <span class="line-through text-gray-400 text-sm">${item['origin_price']}</span>
                    <span class="text-green-400 font-bold ml-2">${item['price']}</span>
                </div>
            """

        html += f'''
        <div class="bg-gray-800 p-3 rounded flex flex-col gap-2">
            <a href="{item['link']}" target="_blank" class="flex items-center gap-4">
              <img src="{item['img']}" onerror="this.src='https://via.placeholder.com/100'" class="w-24 h-24 object-contain bg-white rounded">
              <div>
                <div class="text-lg font-bold text-white">{item['name']}</div>
                <div class="text-xs text-blue-400 font-semibold">{item['source']}</div>
                {price_html}
              </div>
            </a>
            <button class="bg-blue-500 text-white px-2 py-1 text-xs rounded w-fit">加入收藏</button>
        </div>
        '''
    html += '</div>'
    return html

@app.route("/collect", methods=["POST"])
def collect():
    product = {
        "name": request.form.get("name", ""),
        "price": request.form.get("price", ""),
        "desc": request.form.get("desc", ""),
        "img": request.form.get("img", "https://via.placeholder.com/100")
    }
    if not any(p["name"] == product["name"] for p in collection):
        collection.append(product)
        save_collection()
    return render_collection_html()

def render_collection_html():
    html = '<ul class="space-y-2 text-sm">'
    for prod in collection:
        html += f'''
        <li class="flex items-center justify-between group">
            <span>🖥 {prod["name"]} - ${prod["price"]}</span>
            <form hx-post="/remove_item" hx-target="#collection_list" hx-swap="innerHTML">
                <input type="hidden" name="name" value="{prod['name']}">
                <button type="submit" class="ml-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300" title="刪除收藏">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-red-400 hover:text-red-600 transition duration-200" viewBox="0 0 20 20" fill="currentColor">
                      <path fill-rule="evenodd" d="M6 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm3 0a1 1 0 112 0v6a1 1 0 11-2 0V8zm3 0a1 1 0 112 0v6a1 1 0 11-2 0V8zM4 6a1 1 0 011-1h10a1 1 0 011 1v1H4V6zm2-2a2 2 0 012-2h4a2 2 0 012 2v1H6V4z" clip-rule="evenodd" />
                    </svg>
                </button>
            </form>
        </li>
        '''
    html += '</ul>'
    html += '''
    <form hx-post="/clear_collection" hx-target="#collection_list" hx-swap="innerHTML">
        <button type="submit" class="mt-2 text-xs text-red-500 hover:text-red-700 transition duration-200 underline">🧹 清除全部收藏</button>
    </form>
    '''
    return html


def save_collection():
    with open(COLLECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=4)

@app.route("/clear_collection", methods=["POST"])
def clear_collection():
    global collection
    collection = []
    save_collection()
    return render_collection_html()

@app.route("/remove_item", methods=["POST"])
def remove_item():
    name = request.form.get("name", "").strip()
    global collection
    collection = [item for item in collection if item["name"] != name]
    save_collection()
    return render_collection_html()


@app.route("/set_category", methods=["GET"])
def set_category():
    global current_category
    cat = request.args.get("cat", "")
    current_category = cat
    return f'<div class="bg-gray-700 p-3 rounded">已切換到 <strong>{cat}</strong> 類別</div>'

def scroll_to_bottom_script():
    return """
    <script>
      setTimeout(() => {
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      }, 100);
    </script>
    """

# ✅ 電商查詢函式（可重用於未來整合 GPT → 關鍵字 → 查商品）
def search_pchome(keyword, limit=3):
    try:
        url = f"https://ecshweb.pchome.com.tw/search/v3.3/all/results?q={keyword}&page=1&sort=sale/dc"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code != 200:
            return []
        data = res.json()
        items = data.get("prods", [])[:limit]
        return [{
            "name": item.get("name"),
            "price": item.get("price"),
            "origin_price": item.get("originPrice", item.get("price")),
            "img": f"https://a.ecimg.tw{item.get('picB')}",
            "link": f"https://24h.pchome.com.tw/prod/{item['Id']}",
            "source": "PChome"
        } for item in items]
    except Exception as e:
        print("PChome error:", e)
        return []


def search_momo(keyword, limit=3):
    try:
        url = f"https://m.momoshop.com.tw/mosearch/{keyword}.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.text, "html.parser")
        items = []
        for prod in soup.select(".prdListArea li")[:limit]:
            name_tag = prod.select_one(".prdName")
            img_tag = prod.select_one("img")
            link_tag = prod.select_one("a")
            if name_tag and img_tag and link_tag:
                items.append({
                    "name": name_tag.text.strip(),
                    "price": "請點進查看",
                    "origin_price": "請點進查看",
                    "img": img_tag.get("src"),
                    "link": "https://m.momoshop.com.tw" + link_tag.get("href"),
                    "source": "Momo"
                })
        return items
    except Exception as e:
        print("Momo error:", e)
        return []

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            return "❌ 使用者已存在"
        db.session.add(User(username=username, password=password))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user"] = user.username
            return redirect(url_for("index"))
        error = "❌ 登入失敗，帳號或密碼錯誤"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ---------------------------
if __name__ == "__main__":
    if os.path.exists(COLLECTION_FILE):
        with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
            collection = json.load(f)
    app.run(debug=True)