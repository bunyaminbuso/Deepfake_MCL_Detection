import torch
from torch.utils.data import Dataset, DataLoader

class AdvancedMultimodalDataset(Dataset):
    """
    3D Video Kareleri ve Mel-Spektrogram Ses Tensörlerini 
    modelimize besleyen gelişmiş veri yükleyici.
    """
    def __init__(self, num_samples=64, num_frames=16, img_size=(224, 224)):
        self.num_samples = num_samples
        
        # Sentetik 3D Video Tensörleri: (Samples, Channels, Frames, Height, Width)
        self.visual_data = torch.randn(num_samples, 3, num_frames, *img_size)
        
        # Sentetik Mel-Spektrogram Tensörleri: (Samples, Channels, Freq, Time)
        self.audio_data = torch.randn(num_samples, 1, 64, 100)
        
        # Etiketler (0: Gerçek, 1: Fake)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.visual_data[idx], self.audio_data[idx], self.labels[idx]

if __name__ == "__main__":
    dataset = AdvancedMultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    v_batch, a_batch, label_batch = next(iter(dataloader))
    
    print("--- GELİŞMİŞ DATASET TESTİ BAŞARILI ---")
    print(f"Görsel Batch Tensör Boyutu : {v_batch.shape}")
    print(f"Ses Batch Tensör Boyutu    : {a_batch.shape}")