import sys
import os
# Python'ın üst klasördeki modülleri bulabilmesi için yol tanımı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from dataset_loader.dataset import RealVideoDataset
from models.mcl_model import MultimodalContrastiveModel
from models.mcl_loss import MultimodalContrastiveLoss
from utils.augmenter import MultimodalAugmenter

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
    print(f"--- TAM DONANIMLI GÜRÜLTÜ DESTEKLİ VE WANDB ENTEGRELİ EĞİTİM MOTORU [{device}] ---")

    # 1. Weights & Biases Başlatma
    wandb.init(
        project="deepfake-mcl-detection",
        config={
            "learning_rate": 0.0003,
            "architecture": "MCL-3DCNN-ResNet",
            "epochs": 5,
            "batch_size": 2,
            "noise_factor": 0.03
        }
    )

    # 2. Veri Yükleyici ve Gürültü Katmanı
    full_dataset = RealVideoDataset()
    augmenter = MultimodalAugmenter(noise_factor=0.03)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # 3. Model, Loss, Optimizer ve LR Scheduler
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
            
            # Veriye Gürültü Dayanıklılığı (Augmentation) Uygulama
            v_batch_noisy = augmenter.apply_visual_noise(v_batch)
            a_batch_noisy = augmenter.apply_audio_noise(a_batch)

            optimizer.zero_grad()
            v_embed, a_embed = model(v_batch_noisy, a_batch_noisy)
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

        # Learning Rate Takibi
        scheduler.step(avg_val_loss)

        # Metrikleri WandB Canlı Paneline Aktarma
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} - Acc: %{avg_train_acc*100:.1f} "
              f"| Val Loss: {avg_val_loss:.4f} - Acc: %{avg_val_acc*100:.1f}")

        # En İyi Modeli Checkpoint Olarak Kaydetme
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_mcl_model.pt")
            print(f"  --> En iyi model kaydedildi! (Val Loss: {best_val_loss:.4f})")

    wandb.finish()
    print("\n--- EĞİTİM VE WANDB RAPORLAMA TAMAMLANDI ---")

if __name__ == "__main__":
    train_pipeline()