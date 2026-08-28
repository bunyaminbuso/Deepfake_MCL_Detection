import sys
import os
# Python'ın 'utils' ve diğer modülleri bulabilmesi için yol tanımı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
from torch.utils.data import Dataset, DataLoader
from utils.video_processor import VideoAudioProcessor

class RealVideoDataset(Dataset):
    """
    data/real ve data/fake klasörlerindeki MP4 videolarını okuyup
    VideoAudioProcessor üzerinden 3D Yüz Tensörü ve Mel-Spektrograma çeviren veri yükleyici.
    """
    def __init__(self, data_dir="data", max_frames=16):
        self.processor = VideoAudioProcessor(max_frames=max_frames)
        self.samples = []

        real_path = os.path.join(data_dir, "real")
        fake_path = os.path.join(data_dir, "fake")

        # Gerçek Videolar (Etiket: 0)
        if os.path.exists(real_path):
            for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
                for file_path in glob.glob(os.path.join(real_path, ext)):
                    self.samples.append((file_path, 0))

        # Sahte / Deepfake Videolar (Etiket: 1)
        if os.path.exists(fake_path):
            for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
                for file_path in glob.glob(os.path.join(fake_path, ext)):
                    self.samples.append((file_path, 1))

    def __len__(self):
        # Validation hesabının doğru çalışması için sentetik örnek sayısını 8 yaptık
        return len(self.samples) if len(self.samples) > 0 else 8

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            v_tensor = torch.randn(3, 16, 224, 224)
            a_melspec = torch.randn(1, 64, 100)
            label = torch.tensor(0 if idx % 2 == 0 else 1)
            return v_tensor, a_melspec, label

        video_path, label = self.samples[idx]
        v_tensor = self.processor.process_video_frames(video_path)
        _, a_melspec = self.processor.process_audio_signal(video_path)

        return v_tensor, a_melspec, torch.tensor(label)

if __name__ == "__main__":
    dataset = RealVideoDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    v_batch, a_batch, label_batch = next(iter(dataloader))
    
    print("--- GERÇEK VERİ YÜKLEYİCİ PIPELINE'I HAZIR ---")
    print(f"Bulunan Toplam Video Sayısı : {len(dataset.samples)}")
    print(f"İşlenen Görsel Batch Tensörü : {v_batch.shape}")
    print(f"İşlenen Ses Batch Tensörü    : {a_batch.shape}")