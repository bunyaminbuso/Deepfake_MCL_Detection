import torch
from torch.utils.data import DataLoader
from datasets.dataset import MultimodalDataset
from models.mcl_model import MultimodalContrastiveModel
from models.mcl_loss import MultimodalContrastiveLoss

def train():
    # 1. Veri seti, Model, Loss ve Optimizer Tanımlama
    dataset = MultimodalDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MultimodalContrastiveModel()
    criterion = MultimodalContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("--- EĞİTİM DÖNGÜSÜ BAŞLIYOR ---")
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for v_batch, a_batch, _ in dataloader:
            optimizer.zero_grad()
            
            # İleri Besleme (Forward Pass)
            v_embed, a_embed = model(v_batch, a_batch)
            loss = criterion(v_embed, a_embed)
            
            # Geri Yayılım ve Güncelleme (Backward Pass)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] -> Ortalama Kayıp (Loss): {avg_loss:.4f}")

    print("--- EĞİTİM BAŞARIYLA TAMAMLANDI ---")

if __name__ == "__main__":
    train()