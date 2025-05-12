# ai_server/audio_test_main.py
from audio_input.emotion_hubert import predict_emotion_hubert
from audio_input.recorder import record_audio
from audio_input.speech_to_text import speech_to_text

record_audio()
text = speech_to_text("recorded.wav")
emotion = predict_emotion_hubert("recorded.wav")

print(f"\u8bed\u97f3\u8f6c\u6587\u672c\uff1a{text}")
print(f"\u8bc6\u522b\u5230\u60c5\u7eea\uff1a{emotion}")
