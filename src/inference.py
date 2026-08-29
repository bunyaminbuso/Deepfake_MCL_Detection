import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
import torch.nn.functional as F
from models.mcl_model import MultimodalContrastiveModel
from utils.video_processor import VideoAudioProcessor

class DeepfakeDetector:
    def __init__(self, model_path="checkpoints/best_mcl_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VideoAudioProcessor()
        
        self.model = MultimodalContrastiveModel().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"--> Egitilmis model yuklendi: {model_path}")
        else:
            print("--> UYARI: Model baglantisi sifirlandi, varsayilan agirliklar kullanilacak.")
        
        self.model.eval()

    def predict(self, video_path):
        print(f"\n[ANALIZ BASLATILDI] Video: {video_path}")
        
        visual_tensor = self.processor.process_video_frames(video_path).unsqueeze(0).to(self.device)
        _, audio_melspec = self.processor.process_audio_signal(video_path)
        audio_melspec = audio_melspec.unsqueeze(0).to(self.device)

        with torch.no_grad():
            v_embed, a_embed = self.model(visual_tensor, audio_melspec)
            
            v_norm = F.normalize(v_embed, dim=1)
            a_norm = F.normalize(a_embed, dim=1)
            similarity = torch.sum(v_norm * a_norm, dim=1).item()

        # YUKSEK HASSASIYETLI SIGMOIDAL OLCEKLENDIRME
        # Referans esik: 0.35 | Hassasiyet carpani: 12.0
        threshold = 0.35
        scale = 12.0
        logit = (threshold - similarity) * scale
        fake_probability = torch.sigmoid(torch.tensor(logit)).item() * 100.0
        
        is_fake = fake_probability > 50.0

        return {
            "similarity_score": round(similarity, 4),
            "fake_probability": round(fake_probability, 2),
            "verdict": "SAHTE (DEEPFAKE)" if is_fake else "GERCEK (REAL)"
        }

if __name__ == "__main__":
    detector = DeepfakeDetector()
    
    video_files = []
    for folder in ["data/real", "data/fake"]:
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            video_files.extend(glob.glob(os.path.join(folder, ext)))
    
    if len(video_files) == 0:
        print("\n[UYARI] Klasorlerde video bulunamadi.")
    else:
        print(f"\n================ TOPLU TEST BASLATILDI ({len(video_files)} Video) ================")
        for v_path in video_files:
            result = detector.predict(v_path)
            print(f"  └─ Ses-Goruntu Uyum Skoru : {result['similarity_score']}")
            print(f"  └─ Deepfake Olasiligi     : %{result['fake_probability']}")
            print(f"  └─ Nihai Karar            : {result['verdict']}")
        print("\n=======================================================")