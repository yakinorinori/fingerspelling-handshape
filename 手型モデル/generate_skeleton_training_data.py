"""
骨格座標から訓練データセットを生成するスクリプト
ビデオフォルダから骨格特徴量を抽出してnumpyファイルに保存
"""

import os
import glob
import cv2
import numpy as np
import mediapipe as mp
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# クラス定義
HAND_SHAPE_CLASSES = {
    'あ': 1, 'い': 0, 'う': 4, 'え': 1, 'お': 1,
    'か': 4, 'き': 0, 'く': 6, 'け': 6, 'こ': 0,
    'が': 4, 'ぎ': 0, 'ぐ': 6, 'げ': 6, 'ご': 0,
    'さ': 1, 'し': 2, 'す': 2, 'せ': 2, 'そ': 5,
    'ざ': 1, 'じ': 2, 'ず': 2, 'ぜ': 2, 'ぞ': 5,
    'た': 1, 'ち': 0, 'つ': 0, 'て': 6, 'と': 4,
    'だ': 1, 'ぢ': 0, 'づ': 0, 'で': 6, 'ど': 4,
    'な': 4, 'に': 4, 'ぬ': 5, 'ね': 6, 'の': 5,
    'は': 4, 'ひ': 5, 'ふ': 2, 'へ': 0, 'ほ': 6,
    'ば': 4, 'び': 5, 'ぶ': 2, 'べ': 0, 'ぼ': 6,
    'ぱ': 4, 'ぴ': 5, 'ぷ': 2, 'ぺ': 0, 'ぽ': 6,
    'ま': 3, 'み': 3, 'む': 2, 'め': 6, 'や': 0,
    'ゆ': 3, 'よ': 6, 'ら': 4, 'り': 4, 'る': 2,
    'れ': 2, 'ろ': 4, 'わ': 3, 'を': 1, 'ん': 5,
    'ー': 5,
}


class SkeletonFeatureExtractor:
    """手の骨格から特徴量を抽出"""
    
    @staticmethod
    def extract_features_from_frame(hand_landmarks):
        """
        単一フレームから32次元の特徴量を抽出
        """
        if hand_landmarks is None:
            return None
        
        # ランドマーク座標を取得（3次元正規化座標）
        landmarks = [np.array([lm.x, lm.y, lm.z]) for lm in hand_landmarks.landmark]
        
        # 基準点
        p0 = landmarks[0]      # 手首
        p9 = landmarks[9]      # 中指MCP
        
        # 指先
        p4 = landmarks[4]
        p8 = landmarks[8]
        p12 = landmarks[12]
        p16 = landmarks[16]
        p20 = landmarks[20]
        
        # 第一関節
        p3 = landmarks[3]
        p7 = landmarks[7]
        p11 = landmarks[11]
        p15 = landmarks[15]
        p19 = landmarks[19]
        
        # 第二関節
        p2 = landmarks[2]
        p6 = landmarks[6]
        p10 = landmarks[10]
        p14 = landmarks[14]
        p18 = landmarks[18]
        
        # 正規化用の基準線（3次元距離）
        base_line = np.linalg.norm(p9 - p0)
        if base_line < 1e-6:
            base_line = 1.0
        
        # 基準線とx軸のなす角（2D平面での計算）
        base_angle = np.arctan2(p9[1] - p0[1], p9[0] - p0[0])
        
        features = []
        
        # 1-5: 基準点から5本の指先までの距離
        features.append(np.linalg.norm(p4 - p0) / base_line)
        features.append(np.linalg.norm(p8 - p0) / base_line)
        features.append(np.linalg.norm(p12 - p0) / base_line)
        features.append(np.linalg.norm(p16 - p0) / base_line)
        features.append(np.linalg.norm(p20 - p0) / base_line)
        
        # 6-10: 指先と第一関節の差
        features.append((np.linalg.norm(p4 - p0) - np.linalg.norm(p3 - p0)) / base_line)
        features.append((np.linalg.norm(p8 - p0) - np.linalg.norm(p7 - p0)) / base_line)
        features.append((np.linalg.norm(p12 - p0) - np.linalg.norm(p11 - p0)) / base_line)
        features.append((np.linalg.norm(p16 - p0) - np.linalg.norm(p15 - p0)) / base_line)
        features.append((np.linalg.norm(p20 - p0) - np.linalg.norm(p19 - p0)) / base_line)
        
        # 11-14: 親指と他の4指先の距離
        features.append(np.linalg.norm(p4 - p8) / base_line)
        features.append(np.linalg.norm(p4 - p12) / base_line)
        features.append(np.linalg.norm(p4 - p16) / base_line)
        features.append(np.linalg.norm(p4 - p20) / base_line)
        
        # 15-17: 隣同士の指先間距離
        features.append(np.linalg.norm(p8 - p12) / base_line)
        features.append(np.linalg.norm(p12 - p16) / base_line)
        features.append(np.linalg.norm(p16 - p20) / base_line)
        
        # 18-22: 各第二関節と指先の線文とx軸のなす角
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
    
    @staticmethod
    def extract_sequence_from_video(video_path, sample_frames=10):
        """
        ビデオからフレームごとの特徴量を抽出
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        # フレームインデックスを決定
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        features_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                # 手の検出
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    features = SkeletonFeatureExtractor.extract_features_from_frame(
                        results.multi_hand_landmarks[0]
                    )
                    if features is not None:
                        features_list.append(features)
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(features_list) if features_list else None


def generate_training_data(video_dir='../../Video/transformer訓練/手の形状 7クラス分類', 
                          output_dir='skeleton_training_data',
                          sample_frames=1):
    """
    訓練データセットを生成
    フォルダ構造: video_dir/クラス0/文字名/*.mp4
    """
    
    logger.info("\n" + "="*70)
    logger.info("骨格座標訓練データセット生成（3次元距離）")
    logger.info("="*70)
    logger.info(f"ビデオディレクトリ: {video_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # クラス別データを集計
    class_data = defaultdict(list)
    
    # クラスフォルダ（クラス0-6）を処理
    for class_folder in sorted(os.listdir(video_dir)):
        if not class_folder.startswith('クラス'):
            continue
        
        # クラスIDを抽出
        try:
            class_id = int(class_folder.replace('クラス', ''))
        except ValueError:
            continue
        
        class_folder_path = os.path.join(video_dir, class_folder)
        
        if not os.path.isdir(class_folder_path):
            continue
        
        logger.info(f"\n処理中: 【{class_folder}】(クラス{class_id})")
        
        # クラスフォルダ内の文字フォルダを処理
        for character in sorted(os.listdir(class_folder_path)):
            char_dir = os.path.join(class_folder_path, character)
            
            if not os.path.isdir(char_dir):
                continue
            
            # 動画ファイル取得（mp4とmov）
            video_files = sorted(glob.glob(os.path.join(char_dir, '*.mp4')))
            video_files += sorted(glob.glob(os.path.join(char_dir, '*.mov')))
            video_files = sorted(video_files)
            if not video_files:
                continue
            
            logger.info(f"  処理中: 【{character}】- {len(video_files)}個の動画")
            
            for video_idx, video_path in enumerate(video_files, 1):
                video_name = os.path.basename(video_path)
                
                # 特徴量抽出
                features_seq = SkeletonFeatureExtractor.extract_sequence_from_video(
                    video_path,
                    sample_frames=sample_frames
                )
                
                if features_seq is not None:
                    # 各フレームの特徴量を記録
                    for frame_idx, frame_features in enumerate(features_seq):
                        class_data[class_id].append({
                            'features': frame_features,
                            'character': character,
                            'video': video_name,
                            'frame': frame_idx
                        })
    
    # データセットを生成
    logger.info("\n" + "-"*70)
    logger.info("データセット生成中...")
    
    X_all = []
    y_all = []
    
    for class_id in sorted(class_data.keys()):
        data_list = class_data[class_id]
        logger.info(f"クラス{class_id}: {len(data_list)}サンプル")
        
        for data in data_list:
            X_all.append(data['features'])
            y_all.append(class_id)
    
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)
    
    logger.info(f"\n合計サンプル数: {len(X_all)}")
    logger.info(f"特徴量次元: {X_all.shape[1]}")
    
    # ファイルに保存
    output_x = os.path.join(output_dir, 'skeleton_features.npy')
    output_y = os.path.join(output_dir, 'skeleton_labels.npy')
    
    np.save(output_x, X_all)
    np.save(output_y, y_all)
    
    logger.info(f"\nデータを保存しました:")
    logger.info(f"  特徴量: {output_x}")
    logger.info(f"  ラベル: {output_y}")
    
    # 統計情報
    logger.info("\nクラス別サンプル数:")
    for class_id in range(7):
        count = np.sum(y_all == class_id)
        logger.info(f"  クラス{class_id}: {count}個")
    
    logger.info("="*70)
    
    return X_all, y_all


if __name__ == '__main__':
    generate_training_data()
