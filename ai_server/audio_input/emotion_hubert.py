# ai_server/audio_input/emotion_hubert.py
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_DIR = "D:/biyesheji/Models/hubert-base-ch-speech-emotion-recognition"

device = torch.device("cpu")  # 强制使用 CPU

extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

label_map = {
    0: "\u4e2d\u6027",
    1: "\u9ad8\u5174",
    2: "\u6124\u6012",
    3: "\u60b2\u4f24",
    4: "\u60ca\u8bb6",
    5: "\u6050\u60e7"
}

def predict_emotion_hubert(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=1).item()
    return label_map.get(predicted_id, "\u672a\u77e5")
