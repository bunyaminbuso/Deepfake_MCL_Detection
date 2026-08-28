import torch
import torch.nn as nn

class MultimodalContrastiveModel(nn.Module):
    def __init__(self, feature_dim=512, embed_dim=128):
        super(MultimodalContrastiveModel, self).__init__()
        
        # Görsel (Video) Özellik Çıkarıcı
        self.visual_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        # İşitsel (Ses) Özellik Çıkarıcı
        self.audio_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, visual_input, audio_input):
        v_embed = self.visual_encoder(visual_input)
        a_embed = self.audio_encoder(audio_input)
        return v_embed, a_embed

if __name__ == "__main__":
    # Test için sahte veri üreterek modeli deniyoruz
    model = MultimodalContrastiveModel()
    dummy_visual = torch.randn(16, 512)  # 16 örneklik görsel tensor
    dummy_audio = torch.randn(16, 512)   # 16 örneklik ses tensoru

    v_out, a_out = model(dummy_visual, dummy_audio)
    print("--- MODEL TESTİ BAŞARILI ---")
    print(f"Görsel Çıktı Vektörü: {v_out.shape}")
    print(f"Ses Çıktı Vektörü   : {a_out.shape}")