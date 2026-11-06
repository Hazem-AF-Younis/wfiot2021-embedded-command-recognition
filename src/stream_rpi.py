"""Step 4: real-time command detection on a Raspberry Pi 3B+ with a ReSpeaker mic array.

Audio is captured continuously at 48 kHz, decimated to 8 kHz and kept in a
1-second sliding window updated every 0.5 s. MFCCs of the window go to the
TFLite model; if the output is >= 0.5 the command is reported and an LED
on board pin 8 lights (paper, Section V, Fig. 6).

Usage (on the Pi):
    python3 stream_rpi.py --model stop_cnn.tflite --word stop
"""
import argparse
import timeit

import numpy as np
import scipy.signal
import sounddevice as sd
from tflite_runtime.interpreter import Interpreter

from features import NUM_MFCC, SAMPLE_RATE, mfcc

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='wake_word_stop_lite.tflite')
ap.add_argument('--word', default='stop')
ap.add_argument('--threshold', type=float, default=0.5)
ap.add_argument('--mic-rate', type=int, default=48000)
ap.add_argument('--led-pin', type=int, default=8, help='board pin; set 0 to disable GPIO')
ap.add_argument('--debug', action='store_true', help='print score and processing time')
a = ap.parse_args()

REC_DURATION = 0.5
window = np.zeros(int(REC_DURATION * SAMPLE_RATE) * 2)

gpio = None
if a.led_pin:
    import RPi.GPIO as gpio
    gpio.setwarnings(False)
    gpio.setmode(gpio.BOARD)
    gpio.setup(a.led_pin, gpio.OUT, initial=gpio.LOW)

interpreter = Interpreter(a.model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]['index']
out = interpreter.get_output_details()[0]['index']
factor = a.mic_rate // SAMPLE_RATE
assert a.mic_rate % SAMPLE_RATE == 0, 'mic rate must be an integer multiple of 8 kHz'


def callback(rec, frames, time, status):
    if gpio:
        gpio.output(a.led_pin, gpio.LOW)
    start = timeit.default_timer()
    if status:
        print('Error:', status)
    rec = scipy.signal.decimate(np.squeeze(rec), factor)
    window[:len(window) // 2] = window[len(window) // 2:]
    window[len(window) // 2:] = rec

    m = mfcc(window)
    interpreter.set_tensor(inp, np.float32(m.reshape(1, NUM_MFCC, m.shape[1], 1)))
    interpreter.invoke()
    score = interpreter.get_tensor(out)[0][0]
    if score >= a.threshold:
        print(a.word)
        if gpio:
            gpio.output(a.led_pin, gpio.HIGH)
    if a.debug:
        print(f'score={score:.3f}  time={timeit.default_timer() - start:.3f}s')


with sd.InputStream(channels=1, samplerate=a.mic_rate,
                    blocksize=int(a.mic_rate * REC_DURATION), callback=callback):
    print(f'Listening for "{a.word}"... Ctrl+C to stop')
    while True:
        sd.sleep(1000)
