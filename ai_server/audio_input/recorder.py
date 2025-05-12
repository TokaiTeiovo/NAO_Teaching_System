# ai_server/audio_input/recorder.py
import sounddevice as sd
import soundfile as sf

def record_audio(filename='recorded.wav', duration=5, samplerate=16000):
    print("\u5f00\u59cb\u5f55\u97f3...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("\u5f55\u97f3\u5b8c\u6210\uff01\u4fdd\u5b58\u4e3a", filename)