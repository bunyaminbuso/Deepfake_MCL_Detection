import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets.dataset import AdvancedMultimodalDataset
from models.mcl_model import MultimodalContrastiveModel
from models.mcl_loss import MultimodalContrastiveLoss

def calculate_accuracy(v_embed, a_embed, threshold=0.5):
    """
    Ses ve görüntü vektörleri arasındaki Cosine Benzerliğini
    hesaplayarak doğruluk (Accuracy) metriği üretir.
    """
    cos_sim = torch.nn.functional.cosine_similarity(v_embed, a_embed)
    correct = (cos_sim > threshold).float().sum()
    return correct / v_embed.size(0)

def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- EĞİTİM MOTORU BAŞLATILDI [Cihaz: {device}] ---")

    # 1. Veri Setini Train / Validation Olarak Bölme (%80 Eğitim, %20 Doğrulama)
    full_dataset = AdvancedMultimodalDataset(num_samples=100)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 2. Model, Loss, Optimizer ve Scheduler
    model = MultimodalContrastiveModel().to(device)
    criterion = MultimodalContrastiveLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    epochs = 5
    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        # --- EĞİTİM FAZI ---
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for v_batch, a_batch, _ in train_loader:
            v_batch, a_batch = v_batch.to(device), a_batch.to(device)
            
            optimizer.zero_grad()
            v_embed, a_embed = model(v_batch, a_batch)
            loss = criterion(v_embed, a_embed)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(v_embed, a_embed).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # --- DOĞRULAMA (VALIDATION) FAZI ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for v_batch, a_batch, _ in val_loader:
                v_batch, a_batch = v_batch.to(device), a_batch.to(device)
                v_embed, a_embed = model(v_batch, a_batch)
                
                loss = criterion(v_embed, a_embed)
                val_loss += loss.item()
                val_acc += calculate_accuracy(v_embed, a_embed).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        # Öğrenme oranını güncelle
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} - Acc: %{avg_train_acc*100:.1f} "
              f"| Val Loss: {avg_val_loss:.4f} - Acc: %{avg_val_acc*100:.1f}")

        # En iyi modeli Checkpoint olarak kaydet
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_mcl_model.pt")
            print(f"  --> En iyi model kaydedildi! (Val Loss: {best_val_loss:.4f})")

    print("\n--- EĞİTİM VE MODEL KAYIT İŞLEMİ TAMAMLANDI ---")

if __name__ == "__main__":
    train_pipeline()