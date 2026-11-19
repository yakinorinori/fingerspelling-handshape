"""
Webカメラで手の骨格座標を取得してSVMモデルで7クラス分類
特徴量は32次元 × 10フレーム = 320次元を入力
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# クラス定義
CLASS_NAMES = {
    0: "小指伸び(い,き,ち,つ,へ,や,こ)",
    1: "親指のみor全て曲げ(あ,え,お,さ,た,を)",
    2: "親指+人差し(し,す,せ,ふ,む,も,る,れ)",
    3: "人差し+中指+薬指(ま,み,ゆ,わ)",
    4: "人差し+中指(う,か,と,な,に,は,ら,り,ろ)",
    5: "人差し指のみ(ー,そ,ぬ,の,ひ,ん)",
    6: "全て伸びor親指のみ曲(く,け,て,ね,ほ,め,よ)"
}

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class SkeletonFeatureExtractor:
    """手の骨格から32次元の特徴量を抽出"""
    
    def __init__(self):
        self.seq_buffer = []
        self.seq_length = 10
    
    def extract_features(self, hand_landmarks, img_w, img_h):
        """
        MediaPipeの手ランドマークから32次元の特徴量を抽出
        """
        if hand_landmarks is None:
            return None
        
        # ランドマーク座標を取得（正規化座標）
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        # 基準点
        p0 = np.array(landmarks[0][:2])      # 手首
        p9 = np.array(landmarks[9][:2])      # 中指MCP（基準点）
        
        # 5本の指先
        p4 = np.array(landmarks[4][:2])      # 親指先
        p8 = np.array(landmarks[8][:2])      # 人差し指先
        p12 = np.array(landmarks[12][:2])    # 中指先
        p16 = np.array(landmarks[16][:2])    # 薬指先
        p20 = np.array(landmarks[20][:2])    # 小指先
        
        # 第一関節（PIP）
        p3 = np.array(landmarks[3][:2])      # 親指IP
        p7 = np.array(landmarks[7][:2])      # 人差し指PIP
        p11 = np.array(landmarks[11][:2])    # 中指PIP
        p15 = np.array(landmarks[15][:2])    # 薬指PIP
        p19 = np.array(landmarks[19][:2])    # 小指PIP
        
        # 第二関節（MCP）
        p2 = np.array(landmarks[2][:2])      # 親指MCP
        p6 = np.array(landmarks[6][:2])      # 人差し指MCP
        p10 = np.array(landmarks[10][:2])    # 中指MCP
        p14 = np.array(landmarks[14][:2])    # 薬指MCP
        p18 = np.array(landmarks[18][:2])    # 小指MCP
        
        # 正規化用の基準線（点0と点9の距離）
        base_line = np.linalg.norm(p9 - p0)
        if base_line < 1e-6:
            base_line = 1.0
        
        # 基準線とx軸のなす角
        base_angle = np.arctan2(p9[1] - p0[1], p9[0] - p0[0])
        
        features = []
        
        # 1-5: 基準点から5本の指先までの距離（正規化）
        features.append(np.linalg.norm(p4 - p0) / base_line)
        features.append(np.linalg.norm(p8 - p0) / base_line)
        features.append(np.linalg.norm(p12 - p0) / base_line)
        features.append(np.linalg.norm(p16 - p0) / base_line)
        features.append(np.linalg.norm(p20 - p0) / base_line)
        
        # 6-10: 基準点から指先と第一関節の差（正規化）
        features.append((np.linalg.norm(p4 - p0) - np.linalg.norm(p3 - p0)) / base_line)
        features.append((np.linalg.norm(p8 - p0) - np.linalg.norm(p7 - p0)) / base_line)
        features.append((np.linalg.norm(p12 - p0) - np.linalg.norm(p11 - p0)) / base_line)
        features.append((np.linalg.norm(p16 - p0) - np.linalg.norm(p15 - p0)) / base_line)
        features.append((np.linalg.norm(p20 - p0) - np.linalg.norm(p19 - p0)) / base_line)
        
        # 11-14: 親指先と他の4指先の距離（正規化）
        features.append(np.linalg.norm(p4 - p8) / base_line)
        features.append(np.linalg.norm(p4 - p12) / base_line)
        features.append(np.linalg.norm(p4 - p16) / base_line)
        features.append(np.linalg.norm(p4 - p20) / base_line)
        
        # 15-17: 隣同士の指先間距離（正規化）
        features.append(np.linalg.norm(p8 - p12) / base_line)
        features.append(np.linalg.norm(p12 - p16) / base_line)
        features.append(np.linalg.norm(p16 - p20) / base_line)
        
        # 18-22: 各第二関節と指先の線文とx軸のなす角 - 基準線のなす角
        for pj, pi in [(p2, p4), (p6, p8), (p10, p12), (p14, p16), (p18, p20)]:
            angle = np.arctan2(pi[1] - pj[1], pi[0] - pj[0]) - base_angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            features.append(angle)
        
        # 23-27: 各指先と各指の根本との線文とx軸のなす角 - 基準線のなす角
        for pj, pi in [(p3, p4), (p7, p8), (p11, p12), (p15, p16), (p19, p20)]:
            angle = np.arctan2(pi[1] - pj[1], pi[0] - pj[0]) - base_angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            features.append(angle)
        
        # 28-32: 各指先と点0との線文とx軸のなす角 - 基準線のなす角
        for pi in [p4, p8, p12, p16, p20]:
            angle = np.arctan2(pi[1] - p0[1], pi[0] - p0[0]) - base_angle
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            features.append(angle)
        
        return np.array(features, dtype=np.float32)
    
    def add_frame(self, features):
        """フレームをバッファに追加"""
        if features is not None:
            self.seq_buffer.append(features)
            if len(self.seq_buffer) > self.seq_length:
                self.seq_buffer.pop(0)
    
    def get_sequence(self):
        """現在のシーケンスを取得"""
        if len(self.seq_buffer) < self.seq_length:
            return None
        
        return np.array(self.seq_buffer[-self.seq_length:], dtype=np.float32)
    
    def reset(self):
        """バッファをリセット"""
        self.seq_buffer = []


def main():
    """メイン処理"""
    
    logger.info("モデルを読み込み中...")
    
    # SVMモデルと正規化器を読み込む
    model_path = 'skeleton_models/best_skeleton_7class_svm_model.pkl'
    scaler_path = 'skeleton_models/skeleton_svm_scaler.pkl'
    
    if Path(model_path).exists() and Path(scaler_path).exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info(f"モデルを読み込みました: {model_path}")
    else:
        logger.error(f"モデルファイルが見つかりません")
        logger.info("train_skeleton_7class_svm.py で訓練を実行してください")
        return
    
    # 特徴量抽出器
    feature_extractor = SkeletonFeatureExtractor()
    
    # Webカメラを開く
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webカメラを開けません")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    logger.info("Webカメラを起動しました。'q'キーで終了します。")
    logger.info("ウィンドウのXボタンでも終了できます。")
    
    frame_count = 0
    should_exit = False
    
    while not should_exit:
        ret, frame = cap.read()
        if not ret:
            logger.warning("フレーム取得に失敗しました。再試行します...")
            continue
        
        # 画像を反転（鏡像）
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # RGB に変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手の検出
        results = hands.process(rgb_frame)
        
        # 手が検出された場合
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 手を描画
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 特徴量抽出
            features = feature_extractor.extract_features(hand_landmarks, w, h)
            
            if features is not None:
                feature_extractor.add_frame(features)
                
                # シーケンスが揃ったら推論
                sequence = feature_extractor.get_sequence()
                if sequence is not None:
                    # シーケンスを平坦化 (320,)
                    X_flat = sequence.reshape(1, -1)
                    
                    # 正規化
                    X_scaled = scaler.transform(X_flat)
                    
                    # 推論
                    pred_class = model.predict(X_scaled)[0]
                    probas = model.predict_proba(X_scaled)[0]
                    confidence = probas[pred_class]
                    
                    # 結果表示
                    text_class = f"Class: {pred_class}"
                    text_name = f"{CLASS_NAMES[pred_class]}"
                    text_conf = f"Confidence: {confidence:.4f}"
                    
                    # 背景を黒くしてテキストを見やすく
                    cv2.rectangle(frame, (10, 20), (700, 150), (0, 0, 0), -1)
                    cv2.putText(frame, text_class, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, text_name, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, text_conf, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # 各クラスの確率を表示（10フレームごと）
                    if frame_count % 10 == 0:
                        logger.info(f"Frame {frame_count} - Class {pred_class}: {confidence:.4f}")
                        for i in range(7):
                            logger.info(f"  Class {i}: {probas[i]:.4f}")
        else:
            # 手が検出されていない場合
            feature_extractor.reset()
            cv2.putText(frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # フレーム数表示
        cv2.putText(frame, f"Frame: {frame_count}", (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 画面表示
        cv2.imshow('Hand Skeleton Classification (SVM) - 7 Classes', frame)
        
        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 0x1b:
            logger.info("'q'キーが押されました。終了します。")
            should_exit = True
        elif key == ord('r') or key == ord('R'):
            # バッファリセット
            feature_extractor.reset()
            logger.info("シーケンスバッファをリセットしました")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("終了しました")


if __name__ == '__main__':
    main()
