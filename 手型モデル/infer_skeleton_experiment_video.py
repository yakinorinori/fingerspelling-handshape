"""
骨格座標 実験動画の推論スクリプト
複数文字が含まれた動画に対して、フレームごとの7クラス分類結果をCSVに保存
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import logging
import csv
from pathlib import Path

import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 7クラス定義
CLASS_NAMES = {
    0: "小指伸び",
    1: "親指のみor全て曲",
    2: "親指+人差し",
    3: "人差し+中指+薬指",
    4: "人差し+中指",
    5: "人差し指のみ",
    6: "全て伸びor親指のみ曲"
}


class SkeletonTransformerModel(torch.nn.Module):
    """骨格特徴量用Transformerモデル"""
    
    def __init__(self, input_dim=27, num_classes=7, seq_length=10):
        super().__init__()
        
        # 入力投影層
        self.embed_fc = torch.nn.Linear(input_dim, 128)
        self.embed_ln = torch.nn.LayerNorm(128)
        
        # Transformer Encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 分類ヘッド
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * seq_length, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 27)
        batch_size, seq_len, feature_dim = x.shape
        
        # 特徴量投影
        x_embed = self.embed_fc(x)  # (batch, seq_len, 128)
        x_embed = self.embed_ln(x_embed)
        x_embed = torch.relu(x_embed)
        
        # Transformer処理
        x_trans = self.transformer(x_embed)  # (batch, seq_len, 128)
        
        # 分類
        x_flat = x_trans.reshape(batch_size, -1)  # (batch, seq_len*128)
        logits = self.classifier(x_flat)  # (batch, 7)
        
        return logits


class SkeletonFeatureExtractor:
    """手の骨格から特徴量を抽出"""
    
    def __init__(self):
        self.seq_buffer = []
        self.seq_length = 10
    
    def extract_features(self, hand_landmarks, img_w, img_h):
        """
        MediaPipeの手ランドマークから27次元の特徴量を抽出
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
        
        # 第一関節
        p3 = np.array(landmarks[3][:2])      # 親指IP
        p7 = np.array(landmarks[7][:2])      # 人差し指PIP
        p11 = np.array(landmarks[11][:2])    # 中指PIP
        p15 = np.array(landmarks[15][:2])    # 薬指PIP
        p19 = np.array(landmarks[19][:2])    # 小指PIP
        
        # 第二関節
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
        
        # 1-5: 基準点(p0)から5本の指先までの距離（正規化）
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
        
        # 23-27: 各指先と点0の線文とx軸のなす角 - 基準線のなす角
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


def infer_video_frame_by_frame(video_path, model_path='skeleton_models/best_skeleton_7class_model.pth', 
                               output_video_path=None, sample_rate=1):
    """
    ビデオを1フレームごとに処理して7クラス分類
    
    Args:
        video_path: 動画ファイルパス
        model_path: モデルファイルパス
        output_video_path: 出力動画パス（Noneの場合は保存しない）
        sample_rate: フレームサンプリング間隔（1=全フレーム, 5=5フレームごと等）
    
    Returns:
        results: フレーム情報とクラス予測のリスト
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル読み込み
    if not os.path.exists(model_path):
        logger.error(f"モデルが見つかりません: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SkeletonTransformerModel(num_classes=7, seq_length=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # ビデオ読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"ビデオを開けません: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"ビデオ情報: {total_frames}フレーム, FPS: {fps:.2f}, サイズ: {frame_width}x{frame_height}")
    
    # ビデオ出力設定
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        logger.info(f"出力動画: {output_video_path}")
    
    # ランドマーク抽出
    feature_extractor = SkeletonFeatureExtractor()
    
    # フレームごとのランドマーク抽出とフレームバッファ
    all_landmarks = []
    all_frames = []
    frame_idx = 0
    frame_count = 0
    
    logger.info("フレーム抽出中...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                all_landmarks.append((frame_idx, landmarks))
                all_frames.append(frame.copy())
                frame_count += 1
                if frame_count % 50 == 0:
                    logger.info(f"  {frame_count}フレーム処理済み")
        
        frame_idx += 1
    
    cap.release()
    
    if not all_landmarks:
        logger.warning("ランドマークが抽出されませんでした")
        return None
    
    logger.info(f"合計 {len(all_landmarks)} フレーム分のランドマークを抽出")
    
    # 推論結果
    results = []
    seq_length = 10
    
    logger.info("推論・描画処理中...")
    for i, (frame_idx, hand_landmarks) in enumerate(all_landmarks):
        # 特徴量を抽出
        features = feature_extractor.extract_features(hand_landmarks, 0, 0)
        
        frame_to_write = all_frames[i].copy() if i < len(all_frames) else None
        
        if features is not None:
            feature_extractor.add_frame(features)
            
            # シーケンスが揃ったら推論
            sequence = feature_extractor.get_sequence()
            if sequence is not None:
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    confidence, pred_class = torch.max(probs, dim=0)
                
                pred_class_id = pred_class.item()
                confidence = confidence.item()
                
                result = {
                    'frame_idx': int(frame_idx),
                    'class_id': pred_class_id,
                    'class_name': CLASS_NAMES[pred_class_id],
                    'confidence': confidence,
                    'probs': probs.cpu().numpy()
                }
                results.append(result)
                
                # フレームに情報を描画
                if frame_to_write is not None:
                    h, w = frame_to_write.shape[:2]
                    
                    # 背景矩形を描画
                    cv2.rectangle(frame_to_write, (10, 10), (w - 10, 120), (0, 0, 0), -1)
                    
                    # クラス情報を描画
                    text_class = f"Class {pred_class_id}: {CLASS_NAMES[pred_class_id]}"
                    text_conf = f"Confidence: {confidence:.4f}"
                    
                    cv2.putText(frame_to_write, text_class, (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame_to_write, text_conf, (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame_to_write, f"Frame: {frame_idx}", (w - 250, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # 動画に書き込み
                    if out:
                        out.write(frame_to_write)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i + 1}/{len(all_landmarks)} フレーム処理完了")
        else:
            # 特徴量抽出失敗時はバッファリセット
            feature_extractor.reset()
            if out and frame_to_write is not None:
                out.write(frame_to_write)
    
    # 動画ファイルのクローズ
    if out:
        out.release()
        logger.info(f"動画を保存しました: {output_video_path}")
    
    return results


def save_results_to_csv(results, output_csv_path):
    """推論結果をCSVに保存"""
    
    if not results:
        logger.error("保存する結果がありません")
        return False
    
    # 出力ディレクトリ作成
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else '.', exist_ok=True)
    
    # CSVヘッダー
    headers = ['Frame', 'Class_ID', 'Class_Name', 'Confidence']
    for class_id in range(7):
        headers.append(f'Prob_Class{class_id}')
    
    # CSV出力
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for result in results:
            row = [
                result['frame_idx'],
                result['class_id'],
                result['class_name'],
                f"{result['confidence']:.6f}"
            ]
            
            for prob in result['probs']:
                row.append(f"{prob:.6f}")
            
            writer.writerow(row)
    
    logger.info(f"結果を保存しました: {output_csv_path}")
    return True


def process_experiment_folder(folder_path, output_dir='skeleton_experiment_results', sample_rate=5):
    """
    実験フォルダ内の全動画を処理
    
    Args:
        folder_path: 実験フォルダパス
        output_dir: 出力ディレクトリ
        sample_rate: フレームサンプリング間隔
    """
    
    logger.info("\n" + "="*70)
    logger.info("骨格座標 実験動画の推論処理")
    logger.info("="*70)
    logger.info(f"対象フォルダ: {folder_path}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    if not os.path.isdir(folder_path):
        logger.error(f"フォルダが見つかりません: {folder_path}")
        return False
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 動画ファイル検索
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # 原始動画のみを処理（annotated や skeleton_visualized は除外）
    video_files = [f for f in video_files if not ('annotated' in f or 'skeleton_visualized' in f)]
    
    if not video_files:
        logger.error(f"動画ファイルが見つかりません: {folder_path}")
        return False
    
    logger.info(f"見つかった動画: {len(video_files)}個")
    
    # 各動画を処理
    summary_data = []
    
    for video_idx, video_path in enumerate(sorted(video_files), 1):
        video_name = os.path.basename(video_path)
        logger.info(f"\n[{video_idx}/{len(video_files)}] 処理中: {video_name}")
        
        # 出力動画パスを設定
        base_name = os.path.splitext(video_name)[0]
        output_video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
        
        # 推論
        results = infer_video_frame_by_frame(video_path, output_video_path=output_video_path, sample_rate=sample_rate)
        
        if results:
            # CSV保存
            output_csv = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_results.csv")
            save_results_to_csv(results, output_csv)
            
            # サマリー集計
            class_transitions = []
            prev_class = None
            
            for result in results:
                if result['class_id'] != prev_class:
                    class_transitions.append({
                        'frame': result['frame_idx'],
                        'class_id': result['class_id'],
                        'class_name': result['class_name']
                    })
                    prev_class = result['class_id']
            
            summary_data.append({
                'video_name': video_name,
                'total_frames': len(results),
                'class_transitions': len(class_transitions),
                'transition_details': class_transitions
            })
            
            logger.info(f"  クラス遷移: {len(class_transitions)}回")
            for transition in class_transitions[:10]:  # 最初の10個のみ表示
                logger.info(f"    フレーム {transition['frame']}: {transition['class_name']}")
            
            if len(class_transitions) > 10:
                logger.info(f"    ... 他 {len(class_transitions) - 10} 件")
        else:
            logger.warning(f"  推論に失敗しました: {video_name}")
    
    # サマリーCSV生成
    if summary_data:
        summary_csv_path = os.path.join(output_dir, 'summary.csv')
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Video_Name', 'Total_Frames', 'Class_Transitions', 'Transition_Details'])
            
            for data in summary_data:
                details = "; ".join([
                    f"Frame{t['frame']}: {t['class_name']}" 
                    for t in data['transition_details']
                ])
                writer.writerow([
                    data['video_name'],
                    data['total_frames'],
                    data['class_transitions'],
                    details
                ])
        
        logger.info(f"\nサマリーを保存しました: {summary_csv_path}")
    
    logger.info("\n" + "="*70)
    logger.info("処理完了")
    logger.info("="*70)
    
    return True


def main():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        # デフォルト: 実験フォルダ
        folder_path = '../../Video/transformer訓練/実験'
        sample_rate = 5
    
    # 相対パスを絶対パスに変換
    if not os.path.isabs(folder_path):
        folder_path = os.path.abspath(folder_path)
    
    logger.info(f"フォルダパス（絶対）: {folder_path}")
    
    # 処理実行
    process_experiment_folder(
        folder_path,
        output_dir='skeleton_experiment_results',
        sample_rate=sample_rate
    )


if __name__ == '__main__':
    main()
