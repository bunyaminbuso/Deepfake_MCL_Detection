import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, num_samples=100, feature_dim=512):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        # Sahte veri seti simülasyonu (Görsel, Ses ve Etiket)
        self.visual_data = torch.randn(num_samples, feature_dim)
        self.audio_data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 2, (num_samples,)) # 0: Gerçek, 1: Fake

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.visual_data[idx], self.audio_data[idx], self.labels[idx]

if __name__ == "__main__":
    dataset = MultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    v_batch, a_batch, label_batch = next(iter(dataloader))
    print("--- DATASET TESTİ BAŞARILI ---")
    print(f"Görsel Batch Boyutu: {v_batch.shape}")
    print(f"Ses Batch Boyutu   : {a_batch.shape}")
    print(f"Etiket Batch Boyutu: {label_batch.shape}")