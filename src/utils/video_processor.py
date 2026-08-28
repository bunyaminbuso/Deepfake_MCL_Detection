import os
import cv2
import torch
import librosa
import numpy as np

class VideoAudioProcessor:
    def __init__(self, target_size=(224, 224), sample_rate=16000, max_frames=16):
        self.target_size = target_size
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        
        # OpenCV yüz dedektörü yükleme (Hata Korumalı)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def process_video_frames(self, video_path):
        """
        Videoyu okur, kareleri ayırır, yüz bölgesini tespit edip kırpar 
        ve PyTorch Tensörüne (C, T, H, W) dönüştürür.
        """
        if not os.path.exists(video_path):
            return torch.zeros(3, self.max_frames, *self.target_size)

        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Yüz tespiti kontrolü
            if self.face_cascade is not None and getattr(self.face_cascade, 'empty', lambda: True)() == False:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    (x, y, w, h) = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
                    face_crop = frame[y:y+h, x:x+w]
                else:
                    face_crop = frame
            else:
                face_crop = frame

            # Boyutlandırma ve normalizasyon
            resized = cv2.resize(face_crop, self.target_size)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_frame.astype(np.float32) / 255.0
            frames.append(normalized)

        cap.release()

        # Eksik kare padding işlemi
        while len(frames) < self.max_frames:
            frames.append(np.zeros((*self.target_size, 3), dtype=np.float32))

        frames_array = np.array(frames)
        tensor_frames = torch.tensor(frames_array).permute(3, 0, 1, 2)
        return tensor_frames

    def process_audio_signal(self, video_path):
        """
        Videodaki ses izini çıkarır, 16kHz mono sinyale ve Mel-Spektrograma dönüştürür.
        """
        try:
            audio, _ = librosa.load(video_path, sr=self.sample_rate, mono=True)
        except Exception:
            audio = np.zeros(self.sample_rate * 2, dtype=np.float32)

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=64, n_fft=1024, hop_length=512
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        waveform_tensor = torch.tensor(audio).unsqueeze(0)
        melspec_tensor = torch.tensor(log_mel_spec).unsqueeze(0)
        
        return waveform_tensor, melspec_tensor

if __name__ == "__main__":
    processor = VideoAudioProcessor()
    print("--- GERÇEK ÖN İŞLEME PİPELİNE'I HAZIR ---")
    print(f"Hedef Yüz Çözünürlüğü : {processor.target_size}")
    print(f"Ses Örnekleme Hızı    : {processor.sample_rate} Hz")
    print(f"Maksimum Kare Sayısı  : {processor.max_frames}")