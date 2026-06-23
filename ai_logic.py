import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from typing import Optional

class FoodAI:
    """料理画像からカロリーを推定するAIクラス。"""

    # 判定不能と見なす確信度の閾値（%）
    CONFIDENCE_THRESHOLD = 10.0

    # TTA（Test-Time Augmentation）の有効化
    USE_TTA = True

    def __init__(self, model_path: str, csv_path: str):
        self.model_path = model_path
        self.csv_path = csv_path

        # クラス名の読み込み
        try:
            with open("classes.txt", "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            print(f"[OK] {len(self.class_names)} 種類の料理データを読み込みました！")
        except FileNotFoundError:
            print("[ERROR] 'classes.txt' が見つかりません。同じフォルダに保存してください。")
            self.class_names = []

        self.model: Optional[tf.keras.Model] = self._load_model()
        self.df_calo = pd.read_csv(csv_path, encoding="utf-8")

    def _load_model(self) -> Optional[tf.keras.Model]:
        """Kerasモデルを読み込む。失敗時はNoneを返す。"""
        try:
            model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"[OK] モデルを読み込みました: {self.model_path}")
            return model
        except Exception as e:
            print(f"[ERROR] モデルの読み込みに失敗しました: {e}")
            return None

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """PIL画像を(1,224,224,3)のnumpy配列に変換する。"""
        img = pil_image.resize((224, 224))
        return np.array(img) / 255.0

    def _tta_augmentations(self, pil_image: Image.Image) -> list:
        """TTA用の拡張画像リストを生成する（元画像含む8パターン）。"""
        img = pil_image.resize((224, 224))
        variants = [img]

        # 水平反転
        variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))

        # 明るさ調整（少し明るく・少し暗く）
        variants.append(ImageEnhance.Brightness(img).enhance(1.2))
        variants.append(ImageEnhance.Brightness(img).enhance(0.85))

        # コントラスト調整
        variants.append(ImageEnhance.Contrast(img).enhance(1.15))

        # 軽いクロップ（中央85%領域→リサイズ）
        w, h = img.size
        margin = int(w * 0.075)
        cropped = img.crop((margin, margin, w - margin, h - margin)).resize((224, 224))
        variants.append(cropped)
        variants.append(cropped.transpose(Image.FLIP_LEFT_RIGHT))

        # 上下反転（料理画像では稀だが平均化で安定する）
        variants.append(img.rotate(10, resample=Image.BILINEAR, expand=False))

        return variants

    def _get_top3(self, predictions: np.ndarray) -> list:
        """上位3候補を返す。"""
        top3_idx = np.argsort(predictions[0])[::-1][:3]
        results = []
        for idx in top3_idx:
            name = self.class_names[idx] if idx < len(self.class_names) else "不明"
            conf = float(predictions[0][idx] * 100)
            results.append({"name": name, "confidence": conf})
        return results

    def predict(self, pil_image: Image.Image) -> dict:
        """PIL画像を受け取り、料理名・確信度・カロリーを返す。TTAで精度向上。"""
        if self.model is None:
            return {
                "name": "モデル未ロード",
                "confidence": 0.0,
                "calories": "不明",
                "portion": "-",
                "full_name": "不明",
                "top3": []
            }

        if self.USE_TTA:
            # --- TTA: 複数パターンで予測して平均を取る ---
            aug_images = self._tta_augmentations(pil_image)
            batch = np.stack([self._preprocess(img) for img in aug_images], axis=0)
            all_preds = self.model.predict(batch, verbose=0)  # shape: (n_aug, n_classes)
            predictions = np.mean(all_preds, axis=0, keepdims=True)  # 平均アンサンブル
        else:
            img_array = np.expand_dims(self._preprocess(pil_image), axis=0)
            predictions = self.model.predict(img_array, verbose=0)

        idx = int(np.argmax(predictions))
        conf_score = float(predictions[0][idx] * 100)
        top3 = self._get_top3(predictions)

        # 確信度が閾値未満の場合は「判定不能」（上位3候補は返す）
        if conf_score < self.CONFIDENCE_THRESHOLD:
            return {
                "name": "判定不能（未登録）",
                "confidence": conf_score,
                "calories": "不明",
                "portion": "-",
                "full_name": "不明",
                "top3": top3
            }

        # 予測されたインデックスから料理名を取得
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
                "full_name": match.iloc[0]['食品名'],
                "top3": top3
            }
        return {
            "name": detected_item, "confidence": conf_score,
            "calories": "不明", "portion": "不明", "full_name": detected_item,
            "top3": top3
        }