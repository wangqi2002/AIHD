import time
import threading
import pyaudio
import tkinter as tk
import os
import wave
import sounddevice
import json

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from Scripts.audio2action import audio2action
from Scripts.utils_llm import *
from Scripts.utils_ur5e import *


class UI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.root.title("录音")
        self.root.geometry("200x150+630+300")
        self.button = tk.Button(text="录音", font=("Helvetica", 14), width=15, height=5, command=self.click_handler)
        self.button.pack()
        self.label = tk.Label(text="00:00:00",font=("Helvetica", 14),width=10,height=3)
        self.label.pack()
        self.recording = False
        self.recognizing = False

        self.a2a = audio2action()
        self.root.mainloop()

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        # stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=512, input_device_index=11)
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=512)
        frames = []
        start = time.time()
        while self.recording:
            data = stream.read(512)
            frames.append(data)
            passed = time.time() - start
            seconds = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(seconds):02d}")
            self.root.update()

        stream.stop_stream()
        stream.close()
        audio.terminate()

        sound_file = wave.open(f"/home/win/Project/P03/Audio/recording.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(frames))
        sound_file.close()

        audio_path = '/home/win/Project/P03/Audio/recording.wav'
        output_name = '/home/win/Project/P03/Audio/recording1.wav'
        start_time = time.time()
        command = f'ffmpeg -i "{audio_path}" -ar 16000 -y "{output_name}"'
        os.system(command)
        text = self.a2a.audio_to_text(output_name)
        response = self.a2a.action(text)
        print(response)

UI()