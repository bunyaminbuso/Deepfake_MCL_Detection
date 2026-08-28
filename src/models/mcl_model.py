import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Görsel ve işitsel özellikler arasındaki milisaniyelik zamansal uyumu
    ve çelişkileri yakalayan Çapraz Dikkat (Cross-Attention) mekanizması.
    """
    def __init__(self, embed_dim=256, num_heads=4):
        super(CrossModalAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        return self.norm(query + attn_output)

class MultimodalContrastiveModel(nn.Module):
    """
    3D-CNN Görsel Özellik Çıkarıcı + Mel-Spektrogram Ses Çıkarıcı +
    Cross-Attention Katmanı içeren Gelişmiş MCL Mimarisi.
    """
    def __init__(self, embed_dim=256):
        super(MultimodalContrastiveModel, self).__init__()
        
        # 3D-CNN Görsel Kodlayıcı (Zamansal Yüz/Dudak Kareleri)
        self.visual_backbone = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((16, 1, 1))
        )
        self.visual_proj = nn.Linear(64, embed_dim)

        # 2D-CNN İşitsel Kodlayıcı (Mel-Spektrogram)
        self.audio_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 1))
        )
        self.audio_proj = nn.Linear(64, embed_dim)

        # Çapraz Dikkat Etkileşimi (Cross-Modal Fusion)
        self.v_to_a_attention = CrossModalAttention(embed_dim=embed_dim, num_heads=4)
        self.a_to_v_attention = CrossModalAttention(embed_dim=embed_dim, num_heads=4)

        # Nihai Projeksiyon Başlıkları
        self.v_head = nn.Linear(embed_dim, 128)
        self.a_head = nn.Linear(embed_dim, 128)

    def forward(self, visual_frames, audio_melspec):
        # 1. Görsel Özellik Çıkarımı
        v_feat = self.visual_backbone(visual_frames)
        v_feat = v_feat.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        v_proj = self.visual_proj(v_feat)

        # 2. İşitsel Özellik Çıkarımı
        a_feat = self.audio_backbone(audio_melspec)
        a_feat = a_feat.squeeze(-1).permute(0, 2, 1)
        a_proj = self.audio_proj(a_feat)

        # 3. Çapraz Dikkat Entegrasyonu
        v_fused = self.v_to_a_attention(v_proj, a_proj)
        a_fused = self.a_to_v_attention(a_proj, v_proj)

        # 4. Küresel Özet Vektörleri
        v_embed = self.v_head(v_fused.mean(dim=1))
        a_embed = self.a_head(a_fused.mean(dim=1))

        return v_embed, a_embed

if __name__ == "__main__":
    model = MultimodalContrastiveModel()
    
    # 4 Örneklik Test Verisi: (Batch, Channel, Frames/Freq, H/Time, W)
    dummy_visual = torch.randn(4, 3, 16, 224, 224)
    dummy_audio_mel = torch.randn(4, 1, 64, 100)

    v_out, a_out = model(dummy_visual, dummy_audio_mel)
    print("--- GELİŞMİŞ CROSS-ATTENTION MODEL TESTİ BAŞARILI ---")
    print(f"Görsel Embed Çıktı Boyutu : {v_out.shape}")
    print(f"Ses Embed Çıktı Boyutu    : {a_out.shape}")