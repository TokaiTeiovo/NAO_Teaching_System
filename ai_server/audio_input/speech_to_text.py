# ai_server/audio_input/speech_to_text.py
import whisper

def speech_to_text(audio_path):
    model = whisper.load_model("base", device="cpu")  # 精度提升版本
    result = model.transcribe(audio_path, language='zh')
    return result['text']
