# ai_server/audio_input/emotion_from_audio.py
import librosa
import torch
import torch.nn as nn

# 极简 BiLSTM 情感识别模型
torch.manual_seed(42)

class EmotionAudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(40, 32, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64, 4)  # 喜悦、愤怒、悲伤、中性

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, 40]
        _, (h_n, _) = self.lstm(x)
        out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(out)

def extract_mfcc(path, max_len=200):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = torch.nn.functional.pad(torch.tensor(mfcc), (0, pad))
    else:
        mfcc = torch.tensor(mfcc[:, :max_len])
    return mfcc.unsqueeze(0)  # [1, 40, T]

def predict_emotion(audio_path, model_path="audio_emotion_model.pth"):
    model = EmotionAudioNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    mfcc = extract_mfcc(audio_path)
    with torch.no_grad():
        output = model(mfcc)
        predicted = torch.argmax(output, dim=1).item()
    label_map = {0: "喜悦", 1: "愤怒", 2: "悲伤", 3: "中性"}
    return label_map.get(predicted, "未知")
