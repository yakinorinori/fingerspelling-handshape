# fingerspelling-handshape

手の骨格（MediaPipe Hands）32次元特徴量による**手型7クラス分類SVM**の学習と推論。
指文字認識システムの手型認識モジュール。

## クラス定義

| クラス | 手型 | 該当文字 |
|---|---|---|
| 0 | 小指伸び | い,き,ち,つ,へ,や,こ |
| 1 | 親指のみor全て曲げ | あ,え,お,さ,た,を |
| 2 | 親指+人差し | し,す,せ,ふ,む,も,る,れ |
| 3 | 人差し+中指+薬指 | ま,み,ゆ,わ |
| 4 | 人差し+中指 | う,か,と,な,に,は,ら,り,ろ |
| 5 | 人差し指のみ | ー,そ,ぬ,の,ひ,ん |
| 6 | 全て伸びor親指のみ曲 | く,け,て,ね,ほ,め,よ |

## 構成（手型モデル/）

### 学習
- `generate_skeleton_training_data.py` — クラス別動画フォルダ（`クラスN/文字名/*.mp4`）からMediaPipeで32次元骨格特徴量を抽出し `skeleton_training_data/*.npy` を生成
- `train_skeleton_7class_svm.py` — npyを読み込み、10フレームシーケンス（320次元）でSVM（RBFカーネル）を学習。`skeleton_models/best_skeleton_7class_svm_model.pkl` と `skeleton_svm_scaler.pkl` を出力

### 推論（実動画で使用）
- `Mediapipe_skeleton_7class_svm.py` — Webカメラでリアルタイム手型認識
- `infer_skeleton_experiment_video.py` — 複数文字を含む実験動画に対してフレーム毎の7クラス分類結果をCSV出力
- `classify_alphabet_svm.py` — 動画フォルダ（話者/文字.mp4）を一括分類し、動画単位の多数決クラスとクラス分布をCSV出力

## 特徴量（32次元）

手首(p0)と中指MCP(p9)の距離で正規化:
1. 基準点から5指先までの距離（5）
2. 指先と第一関節の距離差（5）
3. 親指と他4指先の距離（4）
4. 隣接指先間の距離（3）
5. 第二関節→指先の角度（5）
6. 第一関節→指先の角度（5）
7. 基準点→指先の角度（5）

学習・推論で同一の特徴量定義を使用（数値一致を検証済み）。

## 使い方

```bash
pip install mediapipe opencv-python numpy scikit-learn joblib

# 1. 学習データ生成（動画フォルダ構造: クラス0〜6/文字名/*.mp4）
python generate_skeleton_training_data.py

# 2. SVM学習
python train_skeleton_7class_svm.py

# 3. 推論
python Mediapipe_skeleton_7class_svm.py        # Webカメラ
python infer_skeleton_experiment_video.py      # 実験動画→CSV
```

## 必要ライブラリ
`mediapipe`, `opencv-python`, `numpy`, `scikit-learn`, `joblib`
