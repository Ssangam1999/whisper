import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import pyaudio
import wave
import uuid
import torch
import torch.nn.functional as F




# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
FORMAT = pyaudio.paInt16
CHANNELS = 1
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS,
                rate = SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_LENGTH)

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_LENGTH
    )

    frames = []
    seconds = 10  # Duration of recording
    print("Start speaking")
    for _ in range(0, int(SAMPLE_RATE / CHUNK_LENGTH * seconds)):
        data = stream.read(CHUNK_LENGTH)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Generate a unique filename
    file_path = f"{uuid.uuid4()}.wav"

    # Save the recorded audio to a file
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print("Audio successfully created:", file_path)
    return file_path

record_audio()