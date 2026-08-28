import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MultimodalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, v_embed, a_embed):
        # Vektörleri normalize et
        v_embed = F.normalize(v_embed, dim=1)
        a_embed = F.normalize(a_embed, dim=1)
        
        # Benzerlik matrisi hesapla (Cosine Similarity)
        logits = torch.matmul(v_embed, a_embed.T) / self.temperature
        labels = torch.arange(v_embed.size(0)).to(v_embed.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

if __name__ == "__main__":
    loss_fn = MultimodalContrastiveLoss()
    v_dummy = torch.randn(16, 128)
    a_dummy = torch.randn(16, 128)
    loss = loss_fn(v_dummy, a_dummy)
    print("--- LOSS TESTİ BAŞARILI ---")
    print(f"Hesaplanan Kayıp Değeri: {loss.item():.4f}")