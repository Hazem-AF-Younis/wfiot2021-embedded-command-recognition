"""Step 1: compute MFCC features for the Google Speech Commands dataset.

Clips that are too short to give a full 16 x 16 MFCC matrix are dropped
(paper, Section IV); longer ones are trimmed to 16 frames.
The data are shuffled and split 80/10/10 into train/validation/test.

Usage:
    python src/extract_features.py --dataset /path/to/speech_commands_v0.02 \
        --out features/all_targets_mfcc_sets.npz
"""
import argparse
import random
from os import listdir
from os.path import isdir, join

import librosa
import numpy as np

from features import LEN_MFCC, SAMPLE_RATE, mfcc


def extract(dataset_path, files, labels, targets):
    xs, ys, dropped = [], [], 0
    for f, y in zip(files, labels):
        if not f.endswith('.wav'):
            continue
        signal, fs = librosa.load(join(dataset_path, targets[int(y)], f), sr=SAMPLE_RATE)
        m = mfcc(signal, fs)
        if m.shape[1] < LEN_MFCC:
            dropped += 1
            continue
        xs.append(m[:, :LEN_MFCC])
        ys.append(y)
    return np.array(xs), np.array(ys), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='unzipped Speech Commands folder')
    ap.add_argument('--out', default='features/all_targets_mfcc_sets.npz')
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--test-ratio', type=float, default=0.1)
    ap.add_argument('--keep', type=float, default=1.0, help='fraction of clips to use')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    targets = sorted(n for n in listdir(a.dataset)
                     if isdir(join(a.dataset, n)) and n != '_background_noise_')
    print('Targets:', targets)

    pairs = [(f, i) for i, t in enumerate(targets) for f in listdir(join(a.dataset, t))]
    random.Random(a.seed).shuffle(pairs)
    pairs = pairs[:int(len(pairs) * a.keep)]
    files, labels = zip(*pairs)

    n_val, n_test = int(len(files) * a.val_ratio), int(len(files) * a.test_ratio)
    splits = {
        'val': (files[:n_val], labels[:n_val]),
        'test': (files[n_val:n_val + n_test], labels[n_val:n_val + n_test]),
        'train': (files[n_val + n_test:], labels[n_val + n_test:]),
    }
    out = {}
    for name, (f, y) in splits.items():
        x, yy, dropped = extract(a.dataset, f, y, targets)
        print(f'{name}: {len(x)} clips kept, {dropped} dropped ({dropped / max(len(f), 1):.1%})')
        out['x_' + name], out['y_' + name] = x, yy

    import os
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out, targets=np.array(targets), **out)
    print('Saved', a.out)


if __name__ == '__main__':
    main()
