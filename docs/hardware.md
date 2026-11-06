# Hardware

| | Raspberry Pi 3 Model B+ (used) | BeagleBone Black (evaluated) |
|---|---|---|
| CPU | 1.4 GHz Cortex-A53 (ARMv8), 4 cores | 1 GHz AM335x Cortex-A8 |
| RAM | 1 GB | 512 MB |
| Audio input | ReSpeaker 4-mic array HAT (12 APA102 LEDs) | needs kernel-level changes for custom audio I/O |

The Raspberry Pi was chosen over the BeagleBone Black for three reasons:

- It has more computing capacity and memory.
- The ReSpeaker plugs straight onto its header.
- It is simpler to set up.

The ReSpeaker LEDs, or an LED on board pin 8 driven by `stream_rpi.py`, show when a detection happens.

## Raspberry Pi setup

1. Install the ReSpeaker driver: [seeed-voicecard](https://github.com/respeaker/seeed-voicecard).
2. Install the Python packages:
   ```bash
   pip3 install tflite-runtime sounddevice scipy python_speech_features RPi.GPIO
   ```
3. Copy `src/features.py`, `src/stream_rpi.py` and a `.tflite` model to the Pi.
4. Run:
   ```bash
   python3 stream_rpi.py --model wake_word_stop_lite.tflite --word stop --debug
   ```

With `--debug`, the script prints each window's score and its processing time, so you can check that it keeps up in real time. Each window must be processed in under 0.5 s.
