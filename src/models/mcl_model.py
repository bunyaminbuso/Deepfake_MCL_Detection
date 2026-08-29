import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultimodalContrastiveModel(nn.Module):
    def __init__(self, embed_dim=128):
        super(MultimodalContrastiveModel, self).__init__()
        
        # 1. Gorusel Encoder: Milyonlarca resimle egitilmis hazir ResNet18 (Transfer Learning)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Ilk katmani 3D/Zaman serisi ve video frame uyumlu hale getiriyoruz
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, embed_dim)
        
        # 2. Ses Encoder: Mel-Spectrogram frekans analizi
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.audio_fc = nn.Linear(64, embed_dim)

    def forward(self, visual_tensor, audio_tensor):
        # visual_tensor shape: [B, C, T, H, W] -> Frame'leri isleme
        b, c, t, h, w = visual_tensor.shape
        # Orta frame'i esas alan gorsel ozellik cikarimi
        middle_frame = visual_tensor[:, :, t // 2, :, :]
        
        v_feat = self.visual_backbone(middle_frame).squeeze(-1).squeeze(-1)
        v_embed = self.visual_fc(v_feat)
        
        # Ses ozellik cikarimi
        a_feat = self.audio_conv(audio_tensor).squeeze(-1).squeeze(-1)
        a_embed = self.audio_fc(a_feat)
        
        # L2 Normalization
        v_embed = F.normalize(v_embed, p=2, dim=1)
        a_embed = F.normalize(a_embed, p=2, dim=1)
        
        return v_embed, a_embed