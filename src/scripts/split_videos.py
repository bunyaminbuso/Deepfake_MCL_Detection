import os
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_dir = os.path.join(BASE_DIR, "data", "real")
fake_dir = os.path.join(BASE_DIR, "data", "fake")

def split_video_into_clips(video_path, output_folder, clip_duration_sec=3):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames_per_clip = fps * clip_duration_sec
    
    filename = os.path.splitext(os.path.basename(video_path))[0]
    
    if "clip_" in filename:
        cap.release()
        return 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    clip_idx = 1
    frame_count = 0
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_clip == 0:
            if out is not None:
                out.release()
            out_path = os.path.join(output_folder, f"{filename}_clip_{clip_idx}.mp4")
            out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
            clip_idx += 1

        if out is not None:
            out.write(frame)
        frame_count += 1

    if out is not None:
        out.release()
    cap.release()
    return clip_idx - 1

if __name__ == "__main__":
    print("\n Uzun insan videoları 3'er saniyelik parçalara bölünüyor...\n")
    total_clips = 0
    for folder in [real_dir, fake_dir]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    v_path = os.path.join(folder, f)
                    count = split_video_into_clips(v_path, folder)
                    if count > 0:
                        print(f"  [BÖLÜNDÜ] {f} -> {count} adet gerçek test videosu çıkarıldı.")
                        total_clips += count

    print(f"\n Toplam {total_clips} adet yeni gerçek yüz klibi hazır!")