import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
from torch.utils.data import Dataset
from utils.video_processor import VideoAudioProcessor

class RealVideoDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.processor = VideoAudioProcessor()
        self.samples = []

        real_videos = glob.glob(os.path.join(data_dir, "real", "*.mp4"))
        fake_videos = glob.glob(os.path.join(data_dir, "fake", "*.mp4"))

        # Gerçek Videolar (Uyumlu Ses-Görüntü)
        for path in real_videos:
            self.samples.append((path, False))
        
        # Sahte Videolar (Uyumsuz Dudak-Ses)
        for path in fake_videos:
            self.samples.append((path, True))

    def __len__(self):
        return len(self.samples) if len(self.samples) > 0 else 4

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            return torch.randn(3, 16, 224, 224), torch.randn(1, 64, 100), torch.tensor(0)

        video_path, is_fake = self.samples[idx]
        v_tensor = self.processor.process_video_frames(video_path)
        _, a_tensor = self.processor.process_audio_signal(video_path)

        # Sahte videolarda ses ile görüntü arasındaki uyumu bozarak tam uyuşmazlık öğretiyoruz
        if is_fake:
            a_tensor = torch.roll(a_tensor, shifts=50, dims=-1)

        label = 1 if is_fake else 0
        return v_tensor, a_tensor, torch.tensor(label)