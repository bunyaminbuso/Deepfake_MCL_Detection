import os
import torch
import torch.nn.functional as F
from models.mcl_model import MultimodalContrastiveModel
from utils.video_processor import VideoAudioProcessor

class DeepfakeDetector:
    def __init__(self, model_path="checkpoints/best_mcl_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VideoAudioProcessor()
        
        # Eğitilmiş Modeli Yükle
        self.model = MultimodalContrastiveModel().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"--> Eğitilmiş model yüklendi: {model_path}")
        else:
            print("--> UYARI: Model ağırlık dosyası bulunamadı, rastgele ağırlıklar kullanılacak.")
        
        self.model.eval()

    def predict(self, video_path):
        """
        Gelen videonun ses ve görüntü uyumunu analiz eder,
        Deepfake/Sahte olma olasılığını ve skorunu üretir.
        """
        print(f"\n[ANALİZ BAŞLATILDI] Video: {video_path}")
        
        # 1. Video ve Ses Ön İşleme
        visual_tensor = self.processor.process_video_frames(video_path).unsqueeze(0).to(self.device)
        _, audio_melspec = self.processor.process_audio_signal(video_path)
        audio_melspec = audio_melspec.unsqueeze(0).to(self.device)

        # 2. İleri Besleme
        with torch.no_grad():
            v_embed, a_embed = self.model(visual_tensor, audio_melspec)
            
            # Normalize ve Cosine Benzerliği Hesaplama
            v_norm = F.normalize(v_embed, dim=1)
            a_norm = F.normalize(a_embed, dim=1)
            similarity = torch.sum(v_norm * a_norm, dim=1).item()

        # Benzerlik skoru düştükçe Deepfake ihtimali artar
        fake_probability = max(0.0, min(100.0, (1.0 - similarity) * 50 + 50))
        is_fake = fake_probability > 50.0

        return {
            "similarity_score": round(similarity, 4),
            "fake_probability": round(fake_probability, 2),
            "verdict": "SAHTE (DEEPFAKE)" if is_fake else "GERÇEK (REAL)"
        }

if __name__ == "__main__":
    detector = DeepfakeDetector()
    # Örnek test çağrısı (Sistemde video yoksa korumalı dummy veriyle çalışır)
    result = detector.predict("data/test_video.mp4")
    
    print("\n--- ANALİZ SONUCU ---")
    print(f"Ses-Görüntü Uyum Skoru : {result['similarity_score']}")
    print(f"Deepfake Olasılığı      : %{result['fake_probability']}")
    print(f"Nihai Karar             : {result['verdict']}")