import tensorflow as tf
import numpy as np
import pandas as pd

class FoodAI:
    def __init__(self, model_path, csv_path):
        self.model_path = model_path
        self.csv_path = csv_path
        
        try:
            with open("classes.txt", "r", encoding="utf-8") as f:
                # 1行ずつ読み込み、改行(\n)や余分な空白を消してリスト化する
                self.class_names = [line.strip() for line in f if line.strip()]
            print(f"[OK] {len(self.class_names)} 種類の料理データを読み込みました！")
        except FileNotFoundError:
            print("[ERROR] エラー: 'classes.txt' が見つかりません。同じフォルダに保存してください。")
            self.class_names = []

        self.model = self._load_model()
        self.df_calo = pd.read_csv(csv_path, encoding="utf-8")

    def _load_model(self):
        return tf.keras.models.load_model(self.model_path, compile=False)

    def predict(self, pil_image):
        # 画像の前処理
        img = pil_image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 予測
        predictions = self.model.predict(img_array, verbose=0)
        idx = np.argmax(predictions)
        conf_score = predictions[0][idx] * 100
        
        # 自信がない（70%未満）場合は「判定不能」にする
        if conf_score < 10.0:
            return {
                "name": "判定不能（未登録）",
                "confidence": conf_score,
                "calories": "不明",
                "portion": "-",
                "full_name": "不明"
            }

        # 予測されたインデックスから料理名を取得
        # ※もしAIが予測した番号(idx)がリストの数を超えていたらエラーを防ぐ
        if idx < len(self.class_names):
            detected_item = self.class_names[idx]
        else:
            detected_item = "不明な料理"

        # CSVからカロリー情報を検索
        match = self.df_calo[self.df_calo['食品名'].str.contains(detected_item, na=False)]
        
        if not match.empty:
            return {
                "name": detected_item,
                "confidence": conf_score,
                "calories": match.iloc[0]['エネルギー (kcal)'],
                "portion": match.iloc[0]['目安量'],
                "full_name": match.iloc[0]['食品名']
            }
        return {
            "name": detected_item, "confidence": conf_score,
            "calories": "不明", "portion": "不明", "full_name": detected_item
        }