import streamlit as st
from PIL import Image
import base64
import time
from ai_logic import FoodAI  # バックエンドを読み込む

# 1. ページ設定 
st.set_page_config(page_title="もぐもぐスキャナー", page_icon="📸", layout="centered")

# --- 背景デザイン設定 (tuika.py/camera.pyのCSSを統合) ---
def set_bg_design(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        base64_image = base64.b64encode(data).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{base64_image}");
                background-size: cover; background-position: center; background-attachment: fixed;
            }}
            .stMain {{
                background: rgba(0, 0, 0, 0.4); backdrop-filter: blur(15px);
                border-radius: 25px; padding: 2rem; margin: 20px 0;
            }}
            h1, h2, h3, p, span, label {{ color: white !important; text-shadow: 2px 2px 8px rgba(0,0,0,0.8); }}
            .food-card {{
                background: rgba(255, 255, 255, 0.15); border-left: 5px solid #00ff88;
                padding: 20px; border-radius: 15px; margin-bottom: 15px;
            }}
            .total-box {{
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                padding: 20px; border-radius: 20px; text-align: center;
                color: #333 !important; font-weight: bold; font-size: 28px;
            }}
            .stButton>button {{
                width: 100%; border-radius: 30px; height: 3.5em;
                background: linear-gradient(135deg, #00dbde 0%, #fc00ff 100%);
                color: white !important; font-weight: bold; border: none;
            }}
            </style>
            """, unsafe_allow_html=True)
    except: pass

set_bg_design("background.png")

# --- AIの準備 ---
@st.cache_resource
def get_ai():
    # food_model_final.h5 があるか確認してください
    return FoodAI(model_path="food_model.keras", csv_path="food_calories.csv")

ai = get_ai()

# --- セッション状態 ---
if 'step' not in st.session_state: st.session_state.step = 'input'
if 'img' not in st.session_state: st.session_state.img = None

# --- メインロジック ---
if st.session_state.step == 'input':
    st.title("🍛 もぐもぐスキャナー")
    st.write("料理を撮影するか、画像をアップロードしてください。")
    
    tab1, tab2 = st.tabs(["📷 カメラで撮影", "📁 画像をアップロード"])
    with tab1:
        cam = st.camera_input("カメラ起動")
        if cam:
            st.session_state.img = Image.open(cam).convert('RGB')
            if st.button("この写真で解析する"):
                st.session_state.step = 'result'; st.rerun()
    with tab2:
        up = st.file_uploader("ファイルを選択", type=["jpg", "png", "jpeg"])
        if up:
            st.session_state.img = Image.open(up).convert('RGB')
            st.image(st.session_state.img, use_column_width=True)
            if st.button("アップロード画像で解析"):
                st.session_state.step = 'result'; st.rerun()

elif st.session_state.step == 'result':
    st.title("📊 解析結果")
    with st.spinner('AIが栄養素をスキャン中...'):
        # ここで本物のAI（バックエンド）を呼び出す
        res = ai.predict(st.session_state.img)
        time.sleep(1) # 演出用の待ち時間

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(st.session_state.img, use_column_width=True)
    
    with col2:
        # tuika.pyのリッチなカードデザインで結果を表示
        if res['confidence'] < 30:
            st.markdown("""
                <div class="food-card">
                    <b>AI判定:</b> <span style="font-size:20px; color:#ff6b6b;">不明</span><br>
                    <small>確信度が低いため判定できませんでした</small>
                </div>
                <div class="total-box">
                    <span style="font-size:16px; color:#555;">推定エネルギー</span><br>
                    不明
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="food-card">
                    🎯 <b>AI判定:</b> <span style="font-size:20px; color:#00ff88;">{res['name']}</span><br>
                    <small>確信度: {res['confidence']:.1f}%</small><br>
                    <hr style="border-color:rgba(255,255,255,0.2);">
                    💡 登録名: {res['full_name']}<br>
                    ⚖️ 目安量: {res['portion']}
                </div>
                <div class="total-box">
                    <span style="font-size:16px; color:#555;">推定エネルギー</span><br>
                    🔥 {res['calories']} kcal
                </div>
            """, unsafe_allow_html=True)

    st.divider()
    if st.button("← 撮り直す"):
        st.session_state.step = 'input'; st.session_state.img = None; st.rerun()