import torch, librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model_name = "garystafford/wav2vec2-deepfake-voice-detector"
model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def predict_audio(audio_path, threshold=0.4):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {"prediction": "fake" if probs[1] >= threshold else "real",
            "real": probs[0].item(), "fake": probs[1].item()}

print(predict_audio("your_real_voice.wav"))
print(predict_audio("some_ai_generated_clip.wav"))