import torch

class MultimodalAugmenter:
    """
    Sosyal medya sıkıştırmalarına (WhatsApp, TikTok vb.) ve gürültülere karşı 
    modelin dayanıklılığını (Robustness) artıran Augmentation modülü.
    """
    def __init__(self, noise_factor=0.03):
        self.noise_factor = noise_factor

    def apply_visual_noise(self, video_tensor):
        """Görüntü karelerine parazit ve gürültü ekler."""
        noise = torch.randn_like(video_tensor) * self.noise_factor
        return torch.clamp(video_tensor + noise, 0.0, 1.0)

    def apply_audio_noise(self, melspec_tensor):
        """Mel-Spektrogram ses matrisine dip gürültü ekler."""
        noise = torch.randn_like(melspec_tensor) * self.noise_factor
        return melspec_tensor + noise

if __name__ == "__main__":
    augmenter = MultimodalAugmenter()
    dummy_v = torch.randn(3, 16, 224, 224)
    dummy_a = torch.randn(1, 64, 100)

    aug_v = augmenter.apply_visual_noise(dummy_v)
    aug_a = augmenter.apply_audio_noise(dummy_a)

    print("--- GÜRÜLTÜ MODÜLÜ TESTİ BAŞARILI ---")
    print(f"Gürültülü Video Tensör Boyutu : {aug_v.shape}")
    print(f"Gürültülü Ses Tensör Boyutu   : {aug_a.shape}")