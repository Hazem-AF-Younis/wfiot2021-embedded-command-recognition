# Challenges in Real-Time Embedded IoT Command Recognition

Code, trained model and paper materials for:

> H. Younis and J. H. L. Hansen, "Challenges in real-time-embedded IoT command recognition," *2021 IEEE 7th World Forum on Internet of Things (WF-IoT)*, New Orleans, LA, USA, 2021, pp. 848–851. doi: [10.1109/WF-IoT51360.2021.9595903](https://doi.org/10.1109/WF-IoT51360.2021.9595903)

CRSS-CI Lab, Center for Robust Speech Systems, The University of Texas at Dallas.

## Overview

The project builds a small speech-command recognizer that runs on a **Raspberry Pi 3 Model B+** with a **ReSpeaker 4-mic array**. Five commands from the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209) are used: *follow, stop, go, learn* and *off*.

- Each command has its own **binary detector** with a sigmoid output.
- Each detector comes in two versions: a **CNN** and a **depthwise-separable CNN (DS-CNN)**.
- Input features are a 16 × 16 MFCC matrix for each 1-second clip at 8 kHz.
- Trained models are converted to **TensorFlow Lite** and run on the Pi.
- On the Pi, a 1-second sliding window is scored every 0.5 s. A score of 0.5 or higher counts as a detection and lights an LED.

```
mic (48 kHz) → decimate to 8 kHz → 1 s sliding window → MFCC (16×16) → TFLite model → p ≥ 0.5 → LED
```

## Results (paper, Table II)

| Command | CNN   | DS-CNN |
|---------|-------|--------|
| stop    | 94.63 | 98.34  |
| learn   | 96.13 | 94.43  |
| go      | 92.89 | 98.80  |
| off     | 97.30 | 97.87  |
| follow  | 93.75 | 93.67  |

Test-set accuracy (%), using a 10% validation and 10% test split.

## Repository layout

```
src/
  features.py          MFCC settings shared by training and inference
  extract_features.py  1. Speech Commands → MFCC train/val/test sets (.npz)
  train.py             2. train a CNN or DS-CNN detector for one command or all five
  convert_tflite.py    3. Keras .h5 → .tflite
  stream_rpi.py        4. real-time detection on the Raspberry Pi + ReSpeaker
models/
  wake_word_stop_model.h5     "stop" CNN detector from the development runs (TF 2.2-era Keras)
  wake_word_stop_lite.tflite  the same model converted to TFLite (input 1×16×16×1)
paper/
  Younis_Hansen_WF-IoT2021_author_manuscript.pdf
  Younis_Hansen_WF-IoT2021_slides.pdf
docs/
  hardware.md          platform notes (Raspberry Pi vs BeagleBone Black, ReSpeaker)
  dataset.md           dataset, commands and speaker-token analysis
```

## Quick start

```bash
pip install -r requirements.txt

# 1. download and unzip Speech Commands v0.02
#    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
python src/extract_features.py --dataset /path/to/speech_commands_v0.02 \
    --out features/all_targets_mfcc_sets.npz

# 2. train detectors for the five commands
python src/train.py --features features/all_targets_mfcc_sets.npz --command all --arch cnn
python src/train.py --features features/all_targets_mfcc_sets.npz --command all --arch dscnn

# 3. convert one model for the Pi
python src/convert_tflite.py models/stop_dscnn.h5 models/stop_dscnn.tflite

# 4. on the Raspberry Pi (pip install tflite-runtime sounddevice scipy python_speech_features RPi.GPIO)
python3 src/stream_rpi.py --model models/stop_dscnn.tflite --word stop
```

The shipped `wake_word_stop_lite.tflite` runs directly with `stream_rpi.py`. The `.h5` file was saved with TF 2.2-era Keras. Load it with TensorFlow ≤ 2.15, or retrain it with `train.py`.

## Notes

- The pipeline started from Shawn Hymel's [TensorFlow Lite speech recognition tutorial](https://github.com/ShawnHymel/tflite-speech-recognition) (Beerware licence).
- The original 2020 scripts were cleaned up for release:
  - Hard-coded lab paths were replaced with command-line arguments.
  - The DS-CNN option was added.
  - A bug was fixed where trimmed clips were appended to the wrong split.
- Retrained accuracies will differ slightly from Table II because of random splits and current TensorFlow versions.
- No audio is stored in this repo. Speech Commands is distributed by Google under CC BY 4.0.
- For the related ARM keyword-spotting baselines (DNN, CNN, DS-CNN, LSTM, GRU and CRNN on microcontrollers), see [ARM-software/ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU).

## Acknowledgment

This work was supported by a grant from Texas Instruments, Dallas, TX.

## Copyright of the paper

The PDF in `paper/` is the authors' manuscript. © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works. The final published version is available at [doi:10.1109/WF-IoT51360.2021.9595903](https://doi.org/10.1109/WF-IoT51360.2021.9595903).

## Citation

```bibtex
@inproceedings{younis2021challenges,
  author    = {Younis, Hazem and Hansen, John H. L.},
  title     = {Challenges in Real-Time-Embedded {IoT} Command Recognition},
  booktitle = {2021 IEEE 7th World Forum on Internet of Things (WF-IoT)},
  pages     = {848--851},
  year      = {2021},
  doi       = {10.1109/WF-IoT51360.2021.9595903}
}
```

## License

Code: MIT (see `LICENSE`). Paper and slides: © 2021 IEEE / the authors, as noted above.
