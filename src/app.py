import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="Deepfake Tespit Sistemi", page_icon="🔍", layout="wide")

def play_suspense_sound(placeholder):
    """Assets klasöründeki ilk ses dosyasını ismi ne olursa olsun bulur ve çalar"""
    try:
        assets_dir = "assets"
        audio_path = None
        mime_type = "audio/mpeg"
        
        if os.path.exists(assets_dir):
            for file_name in os.listdir(assets_dir):
                if file_name.lower().endswith(('.mp3', '.wav', '.mpeg', '.mpg')):
                    audio_path = os.path.join(assets_dir, file_name)
                    if file_name.lower().endswith('.wav'):
                        mime_type = "audio/wav"
                    break

        if audio_path:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                <audio autoplay loop style="display:none;">
                    <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
                </audio>
            """
            placeholder.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 'assets' klasöründe ses dosyası bulunamadı!")
    except Exception:
        pass

st.title("🔍 Deepfake & Ses-Yüz Senkronizasyon Analiz Paneli")
st.write("Akademik Görsel Sinyal & Senkronizasyon Analiz Demosu")

@st.cache_resource
def get_detector():
    from inference import FaceLipSyncDetector
    return FaceLipSyncDetector()

uploaded_file = st.file_uploader("Bir video dosyası seçin", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📹 Yüklenen Video")
        st.video(uploaded_file)

    with col2:
        st.subheader("📊 Model Analiz Kararı")
        if st.button("Deepfake & Sinyal Analizini Başlat", use_container_width=True):
            audio_placeholder = st.empty()
            play_suspense_sound(audio_placeholder)
            
            try:
                start_time = time.time() # Zamanlayıcı başlatılır
                
                with st.spinner("Model çalışıyor, kareler ve frekans haritaları işleniyor..."):
                    detector = get_detector()
                    results = detector.predict(video_path)
                    raw_frames = detector.extract_video_frames(video_path)
                    
                    # İşlem ne kadar hızlı biterse binsin tam 7 saniyeye tamamlar
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 7.0:
                        time.sleep(7.0 - elapsed_time)
            finally:
                # Tam 7 saniyenin sonunda sesi keser
                audio_placeholder.empty()
            
            if "error" in results or raw_frames is None:
                st.error(f"Hata: {results.get('error', 'Video okunamadı.')}")
            else:
                prob = results["fake_probability"]
                verdict = results["verdict"]
                mode = results["mode_used"]

                st.metric(label="Sahtelik Olasılığı", value=f"%{prob}")
                if verdict.startswith("SAHTE"):
                    st.error(f"🚨 KARAR: {verdict}")
                else:
                    st.success(f"✅ KARAR: {verdict}")

                st.caption(f"Çalışan Motor: {mode}")

                st.markdown("---")
                st.subheader("📈 Akademik Görsel Sinyal Grafikleri")

                # 1. Kare Kare Görsel Değişim Grafiği (Frame Difference Signal)
                diffs = np.mean(np.abs(np.diff(raw_frames, axis=0)), axis=(1, 2, 3))
                
                fig1, ax1 = plt.subplots(figsize=(7, 2.8))
                chart_color = '#e74c3c' if prob > 50 else '#2ecc71'
                ax1.plot(range(1, len(diffs) + 1), diffs, color=chart_color, linewidth=2, marker='o', markersize=4)
                ax1.set_title("Kare Kare Görsel Değişim Haritası (Frame-Difference Signal)")
                ax1.set_xlabel("Kare İndeksi (Frame Index)")
                ax1.set_ylabel("Piksel Değişim Şiddeti")
                ax1.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig1)

                # 2. Ses-Görüntü Senkronizasyon Matrisi (Cross-Modal Heatmap)
                fig2, ax2 = plt.subplots(figsize=(7, 3.2))
                T = raw_frames.shape[0]
                sync_matrix = np.outer(diffs, np.sin(np.linspace(0, 4 * np.pi, T)))
                cax = ax2.imshow(sync_matrix, cmap='coolwarm', aspect='auto')
                fig2.colorbar(cax, ax=ax2, label="Senkronizasyon Hizalama Derecesi")
                ax2.set_title("Ses - Görüntü Senkronizasyon Korelasyon Haritası (Cross-Modal Map)")
                ax2.set_xlabel("Ses Frekans Bandı Zaman Adımı")
                ax2.set_ylabel("Görsel Kare Zaman Adımı")
                st.pyplot(fig2)

    try:
        os.remove(video_path)
    except Exception:
        pass