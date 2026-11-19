# -*- coding: utf-8 -*-
"""
makeVideo/output の話者A〜Eのアルファベット動画を
手型SVM（7クラス）でフレーム毎に認識し、動画単位でクラス分布を集計する。

出力: ~/Desktop/svm_alphabet_results/
    - frame_predictions.csv  (フレーム毎の予測)
    - summary.csv            (動画毎の多数決クラスと分布)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

SKELETON_DIR = Path(__file__).resolve().parent.parent / 'Fingerspelling-word-division' / 'skeleton'
sys.path.insert(0, str(SKELETON_DIR))

import cv2  # noqa: E402
from integrated_recognition_boundary_detection import HandShapeSVMRecognizer  # noqa: E402

CLASS_NAMES = {
    0: "小指伸び(い,き,ち,つ,へ,や,こ)",
    1: "親指のみor全て曲げ(あ,え,お,さ,た,を)",
    2: "親指+人差し(し,す,せ,ふ,む,も,る,れ)",
    3: "人差し+中指+薬指(ま,み,ゆ,わ)",
    4: "人差し+中指(う,か,と,な,に,は,ら,り,ろ)",
    5: "人差し指のみ(ー,そ,ぬ,の,ひ,ん)",
    6: "全て伸びor親指のみ曲(く,け,て,ね,ほ,め,よ)",
}

VIDEO_ROOT = Path(__file__).resolve().parent / 'output'
OUT_DIR = Path.home() / 'Desktop' / 'svm_alphabet_results'


def main():
    recognizer = HandShapeSVMRecognizer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    summary_rows = []

    videos = sorted(VIDEO_ROOT.glob('*/[a-z]*.mp4'))
    print(f'処理対象: {len(videos)} 本')

    for vi, video in enumerate(videos, 1):
        speaker = video.parent.name
        letter = video.stem
        recognizer.reset()

        cap = cv2.VideoCapture(str(video))
        preds = []
        confs = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cls, conf = recognizer.predict(frame)
            if cls is not None:
                cls = int(cls)
                preds.append(cls)
                confs.append(float(conf))
                frame_rows.append([speaker, letter, frame_idx, cls, f'{conf:.4f}'])
            else:
                frame_rows.append([speaker, letter, frame_idx, -1, ''])
            frame_idx += 1
        cap.release()

        if preds:
            counter = Counter(preds)
            majority, maj_count = counter.most_common(1)[0]
            dist = {c: counter.get(c, 0) / len(preds) for c in range(7)}
            mean_conf = sum(confs) / len(confs)
            summary_rows.append({
                'speaker': speaker, 'letter': letter,
                'total_frames': frame_idx, 'detected_frames': len(preds),
                'majority_class': majority,
                'majority_class_name': CLASS_NAMES[majority],
                'majority_ratio': f'{maj_count / len(preds):.3f}',
                'mean_confidence': f'{mean_conf:.3f}',
                **{f'class{c}_ratio': f'{dist[c]:.3f}' for c in range(7)},
            })
        else:
            summary_rows.append({
                'speaker': speaker, 'letter': letter,
                'total_frames': frame_idx, 'detected_frames': 0,
                'majority_class': -1, 'majority_class_name': '手が検出されず',
                'majority_ratio': '', 'mean_confidence': '',
                **{f'class{c}_ratio': '' for c in range(7)},
            })
        print(f'[{vi}/{len(videos)}] {speaker}/{letter}: '
              f'{summary_rows[-1]["majority_class_name"]} '
              f'(検出 {len(preds)}/{frame_idx} フレーム)')

    # フレーム毎CSV
    with open(OUT_DIR / 'frame_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['speaker', 'letter', 'frame', 'pred_class', 'confidence'])
        w.writerows(frame_rows)

    # サマリCSV
    with open(OUT_DIR / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f'\n結果保存先: {OUT_DIR}')


if __name__ == '__main__':
    main()
