import os
import tensorflow as tf
from tensorflow.keras import layers, models

# 🌟 RTX 3060 が認識されているかチェック！
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🔥 最高です！GPU（RTX 3060）を検出しました: {gpus}")
else:
    print("⚠️ 警告: GPUが検出されていません。CPUで学習するため非常に時間がかかります。")

# 1. データセットのパス（先ほどPCに置いたフォルダを指定します）
# 例: 'data/UECFOOD100' など、あなたの環境に合わせて書き換えてください
データディレクトリ = './data/UECFOOD100'

# 2. クラスIDの自動取得とソート
try:
    クラスIDリスト = [d for d in os.listdir(データディレクトリ)
                     if os.path.isdir(os.path.join(データディレクトリ, d)) and not d.startswith('.')]
    クラスIDリスト = sorted(クラスIDリスト, key=int)
except FileNotFoundError:
    print(f"❌ エラー: '{データディレクトリ}' フォルダが見つかりません！パスを確認してください。")
    exit()

クラス数 = len(クラスIDリスト)
print(f"📁 ディレクトリ内に合計 {クラス数} 個のクラスを検出しました。")

# 3. ハイパーパラメータの設定
画像サイズ = (224, 224)
バッチサイズ = 32

# 4. データセットの読み込み
訓練データ = tf.keras.utils.image_dataset_from_directory(
    データディレクトリ,
    label_mode='categorical',
    class_names=クラスIDリスト,
    image_size=画像サイズ,
    batch_size=バッチサイズ,
    validation_split=0.2,
    subset="training",
    seed=123
)

検証データ = tf.keras.utils.image_dataset_from_directory(
    データディレクトリ,
    label_mode='categorical',
    class_names=クラスIDリスト,
    image_size=画像サイズ,
    batch_size=バッチサイズ,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# 5. 前処理とパフォーマンス最適化
正規化レイヤー = layers.Rescaling(1./255)
訓練データ = 訓練データ.map(lambda x, y: (正規化レイヤー(x), y))
検証データ = 検証データ.map(lambda x, y: (正規化レイヤー(x), y))

訓練データ = 訓練データ.prefetch(buffer_size=tf.data.AUTOTUNE)
検証データ = 検証データ.prefetch(buffer_size=tf.data.AUTOTUNE)

# データ拡張（画像を自動で少し反転・回転・ズーム）
データ拡張 = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])

# 6. ファインチューニング用モデルの構築
print("🧠 ベースモデル（MobileNetV2）をダウンロード＆読み込み中...")
ベースモデル = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# 🌟 ファインチューニングの設定
ベースモデル.trainable = True

fine_tune_at = 100
for layer in ベースモデル.layers[:fine_tune_at]:
    layer.trainable = False

print(f"🔓 MobileNetV2の {fine_tune_at} 層目以降の封印を解除しました！")

モデル = models.Sequential([
    データ拡張,
    ベースモデル,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(クラス数, activation='softmax')
])

# 7. コンパイルと学習の開始
モデル.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print(f"🚀 {クラス数} 種の料理識別モデルのローカル学習を開始します...")
# PCのフルパワーを使うので Epochs 30 で回します
履歴 = モデル.fit(訓練データ, validation_data=検証データ, epochs=30)

# 8. 学習済みモデルの保存（アプリと同じ場所に直接保存されます！）
保存パス = 'food_model_108_finetuned.h5'
モデル.save(保存パス)
print(f"🎉 学習完了！モデルを保存しました: {保存パス}")