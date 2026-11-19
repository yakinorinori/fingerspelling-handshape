"""
SVMで7クラス手の形状分類を訓練
特徴量は32次元 × 10フレーム = 320次元を入力
"""

import numpy as np
import logging
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


class SkeletonSequenceDataset:
    """骨格シーケンスデータセット"""
    
    def __init__(self, features, labels, seq_length=10):
        """
        features: (N, 32) - 特徴量
        labels: (N,) - ラベル
        seq_length: シーケンス長
        """
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        
        # シーケンスを作成
        self.sequences = []
        self.seq_labels = []
        
        for i in range(len(features) - seq_length + 1):
            seq = features[i:i + seq_length]  # (seq_length, 32)
            label = labels[i + seq_length - 1]  # 最後のフレームのラベルを使用
            
            self.sequences.append(seq)
            self.seq_labels.append(label)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)  # (num_seq, seq_length, 32)
        self.seq_labels = np.array(self.seq_labels, dtype=np.int64)  # (num_seq,)
    
    def flatten_sequences(self):
        """シーケンスを平坦化 (num_seq, seq_length*32)"""
        num_seq = self.sequences.shape[0]
        flattened = self.sequences.reshape(num_seq, -1)  # (num_seq, 320)
        return flattened, self.seq_labels


def train_svm_model():
    """SVMモデルを訓練"""
    
    logger.info("=" * 70)
    logger.info("骨格座標SVM訓練")
    logger.info("=" * 70)
    
    # データを読み込む
    logger.info("\nデータを読み込み中...")
    features = np.load('skeleton_training_data/skeleton_features.npy')
    labels = np.load('skeleton_training_data/skeleton_labels.npy')
    
    logger.info(f"  特徴量形状: {features.shape}")
    logger.info(f"  ラベル形状: {labels.shape}")
    
    # シーケンスデータセットを作成
    seq_length = 10
    dataset = SkeletonSequenceDataset(features, labels, seq_length=seq_length)
    X_flat, y = dataset.flatten_sequences()
    
    logger.info(f"  シーケンス数: {X_flat.shape[0]}")
    logger.info(f"  入力次元: {X_flat.shape[1]} (32 × {seq_length})")
    logger.info(f"  訓練セット: {X_flat.shape[0]}サンプル（全データを使用）")
    
    # クラス分布
    logger.info("\nクラス別サンプル数:")
    for cls_id in range(7):
        count = np.sum(y == cls_id)
        logger.info(f"  クラス{cls_id}: {count}個")
    
    # 特徴量の正規化
    logger.info("\n特徴量を正規化中...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    logger.info("\n" + "-" * 70)
    logger.info("SVMモデル訓練開始")
    logger.info("-" * 70)
    
    # SVMモデルの訓練
    start_time = time.time()
    
    model = SVC(
        kernel='rbf',          # RBFカーネル
        C=1.0,                 # 正則化パラメータ
        gamma='scale',         # カーネルの係数
        probability=True,      # 確率推定を有効化
        verbose=1
    )
    
    logger.info("\nモデル訓練中...")
    model.fit(X_scaled, y)
    
    elapsed_time = time.time() - start_time
    logger.info(f"訓練完了 (所要時間: {elapsed_time:.2f}秒)")
    
    # 訓練精度
    train_pred = model.predict(X_scaled)
    train_acc = accuracy_score(y, train_pred)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"訓練精度: {train_acc:.4f}")
    logger.info("=" * 70)
    
    # 混同行列
    logger.info("\n混同行列:")
    cm = confusion_matrix(y, train_pred)
    logger.info("\n" + str(cm))
    
    # 分類レポート
    logger.info("\n分類レポート:")
    logger.info("\n" + classification_report(y, train_pred, target_names=[f"Class {i}" for i in range(7)]))
    
    # モデルを保存
    model_dir = Path('skeleton_models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'best_skeleton_7class_svm_model.pkl'
    scaler_path = model_dir / 'skeleton_svm_scaler.pkl'
    
    joblib.dump(model, str(model_path))
    joblib.dump(scaler, str(scaler_path))
    
    logger.info(f"\nモデルを保存しました: {model_path}")
    logger.info(f"正規化器を保存しました: {scaler_path}")
    logger.info("\n訓練完了")


if __name__ == '__main__':
    train_svm_model()
