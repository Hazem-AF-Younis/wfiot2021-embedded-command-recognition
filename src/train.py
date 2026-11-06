"""Step 2: train one binary command detector (CNN or DS-CNN) per command.

Each model answers "is this clip the target command?" with a sigmoid output,
trained with binary cross-entropy and RMSprop (paper, Section IV).

Usage:
    python src/train.py --features features/all_targets_mfcc_sets.npz --command stop --arch cnn
    python src/train.py --features features/all_targets_mfcc_sets.npz --command all --arch dscnn
"""
import argparse
import os

import numpy as np
from tensorflow.keras import layers, models

PAPER_COMMANDS = ['follow', 'stop', 'go', 'learn', 'off']


def build(arch, input_shape):
    Conv = layers.Conv2D if arch == 'cnn' else layers.SeparableConv2D
    m = models.Sequential([layers.Input(shape=input_shape)])
    for filters in (32, 32, 64):
        m.add(Conv(filters, (2, 2), activation='relu'))
        m.add(layers.MaxPooling2D(pool_size=(2, 2)))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation='relu'))
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return m


def train_one(fs, command, arch, epochs, outdir):
    targets = list(fs['targets'])
    idx = targets.index(command)
    x = {k: fs['x_' + k][..., np.newaxis] for k in ('train', 'val', 'test')}
    y = {k: (fs['y_' + k] == idx).astype('float32') for k in ('train', 'val', 'test')}

    model = build(arch, x['train'].shape[1:])
    model.summary()
    model.fit(x['train'], y['train'], epochs=epochs, batch_size=100,
              validation_data=(x['val'], y['val']))
    loss, acc = model.evaluate(x['test'], y['test'])
    path = os.path.join(outdir, f'{command}_{arch}.h5')
    model.save(path)
    print(f'{command} / {arch}: test accuracy {acc:.4f} -> {path}')
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--command', default='stop', help="command word, or 'all' for the paper's five")
    ap.add_argument('--arch', choices=['cnn', 'dscnn'], default='cnn')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--outdir', default='models')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    fs = np.load(a.features)
    commands = PAPER_COMMANDS if a.command == 'all' else [a.command]
    results = {c: train_one(fs, c, a.arch, a.epochs, a.outdir) for c in commands}
    for c, acc in results.items():
        print(f'{c:8s} {a.arch:6s} {acc:.2%}')


if __name__ == '__main__':
    main()
