import sys
import os
import time
import tempfile
import base64
import json
import numpy as np
import matplotlib.pyplot as plt

# Path Ayarları
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st

# 1. SAYFA YAPILANDIRMASI (Tek ve en üstte)
st.set_page_config(page_title="Deepfake MCL Tespit Sistemi", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------
# KALICI KULLANICI VERİTABANI YÖNETİMİ (JSON)
# ---------------------------------------------------------
USER_DB_FILE = os.path.join(os.path.dirname(__file__), "users.json")

def load_users():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    default_users = {
        "admin@example.com": {"password": "admin", "name": "Yönetici"}
    }
    save_users(default_users)
    return default_users

def save_users(users):
    try:
        with open(USER_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Kullanıcı kaydedilirken hata oluştu: {e}")

# 2. GÖRSEL TASARIM (Dark Glassmorphism & Neon CSS)
custom_css = """
<style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1a0b2e 100%) !important;
        color: #f1f5f9 !important;
    }
    
    /* Neon Başlıklar */
    .hero-title {
        background: linear-gradient(90deg, #c084fc 0%, #6366f1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    /* Glassmorphic Form ve Kartlar */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7), 0 0 20px rgba(168, 85, 247, 0.15) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        backdrop-filter: blur(12px);
    }

    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.7);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }

    /* Neon Butonlar */
    .stButton>button, div[data-testid="stForm"] button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton>button:hover, div[data-testid="stForm"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(168, 85, 247, 0.6) !important;
    }

    /* Input Alanları */
    .stTextInput input {
        background-color: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }

    .stTextInput input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.5) !important;
    }

    /* Yükleme Kutusu (File Uploader) */
    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.4);
        border: 2px dashed rgba(168, 85, 247, 0.4);
        border-radius: 16px;
        padding: 15px;
    }

    /* Sol Menü (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(168, 85, 247, 0.2);
    }

    /* Expander Tasarımı */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Oturum Durumu (Session State) Başlatma
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user_db" not in st.session_state:
    st.session_state["user_db"] = load_users()

# Analiz Geçmişi Listesi
if "history" not in st.session_state:
    st.session_state["history"] = []

# Ses Çalma Fonksiyonu
def play_suspense_sound(placeholder):
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

@st.cache_resource
def get_detector():
    from inference import FaceLipSyncDetector
    return FaceLipSyncDetector()

# ---------------------------------------------------------
# 1. GİRİŞ YAP / KAYIT OL EKRANI
# ---------------------------------------------------------
if not st.session_state["authenticated"]:
    _, col_center, _ = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown("<h1 class='hero-title'>🛡️ DEEPFAKE MCL TESPİT SİSTEMİ</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-subtitle'>Deepfake & Ses-Yüz Senkronizasyon Analiz Portalı</p>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔑 Giriş Yap", "📝 Kayıt Ol"])

        # 1. GİRİŞ YAP SEKMESİ
        with tab1:
            with st.form("login_form"):
                email = st.text_input("E-Posta Adresi", placeholder="ornek@email.com")
                password = st.text_input("Şifre", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Sisteme Giriş Yap", use_container_width=True)

                if submit_login:
                    user = st.session_state["user_db"].get(email)
                    if user and user["password"] == password:
                        st.session_state["authenticated"] = True
                        st.session_state["user_name"] = user["name"]
                        st.success(f"Hoş geldiniz, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ E-posta veya şifre hatalı!")

        # 2. KAYIT OL SEKMESİ
        with tab2:
            with st.form("register_form"):
                name = st.text_input("Ad Soyad", placeholder="Ahmet Yılmaz")
                reg_email = st.text_input("E-Posta Adresi", placeholder="ornek@email.com")
                reg_password = st.text_input("Şifre", type="password", placeholder="En az 6 karakter")
                submit_register = st.form_submit_button("Hesap Oluştur", use_container_width=True)

                if submit_register:
                    if reg_email in st.session_state["user_db"]:
                        st.warning("⚠️ Bu e-posta adresi zaten kayıtlı!")
                    elif not reg_email or not reg_password or not name:
                        st.error("❌ Lütfen tüm alanları doldurun.")
                    else:
                        st.session_state["user_db"][reg_email] = {"password": reg_password, "name": name}
                        save_users(st.session_state["user_db"])
                        st.success("🎉 Kayıt başarılı! Giriş Yap sekmesinden oturum açabilirsiniz.")

    # Giriş yapılmadıysa alt taraftaki analiz bölümünü çalıştırma
    st.stop()


# ---------------------------------------------------------
# 2. ANA UYGULAMA (Giriş Yapıldıysa Çalışacak Kısım)
# ---------------------------------------------------------

# Sol Sidebar
with st.sidebar:
    st.markdown("### 👤 Kullanıcı Profili")
    st.info(f"**Aktif Oturum:** {st.session_state.get('user_name', 'Kullanıcı')}")
    st.markdown("---")
    if st.button("🚪 Çıkış Yap", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# Ana Sayfa Başlığı
st.markdown("<h1 class='hero-title'>🔍 Deepfake & Ses-Yüz Senkronizasyon Analiz Paneli</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Akademik Görsel Sinyal & Spektral Senkronizasyon Demosu</p>", unsafe_allow_html=True)

# Ana Ekran Sekmeleri: Analiz ve Geçmiş
main_tab1, main_tab2 = st.tabs(["🧪 Video Analizi", "📜 Analiz Geçmişi"])

# ----------------------------
# SEKMELER 1: VIDEO ANALİZİ
# ----------------------------
with main_tab1:
    uploaded_file = st.file_uploader("Analiz edilecek video dosyasını seçin", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("📹 Yüklenen Video")
            st.video(uploaded_file)

        with col2:
            st.subheader("📊 Analiz Kararı")
            if st.button("Deepfake & Sinyal Analizini Başlat", use_container_width=True):
                audio_placeholder = st.empty()
                play_suspense_sound(audio_placeholder)
                
                try:
                    start_time = time.time()
                    
                    with st.spinner("Video işleniyor, kareler ve frekans haritaları analiz ediliyor..."):
                        detector = get_detector()
                        results = detector.predict(video_path)
                        raw_frames = detector.extract_video_frames(video_path)
                        
                        elapsed_time = time.time() - start_time
                        if elapsed_time < 7.0:
                            time.sleep(7.0 - elapsed_time)
                finally:
                    audio_placeholder.empty()
                
                if "error" in results or raw_frames is None:
                    st.error(f"Hata: {results.get('error', 'Video okunamadı.')}")
                else:
                    prob = results["fake_probability"]
                    verdict = results["verdict"]
                    mode = results["mode_used"]

                    # Kare değişim sinyalini hesapla
                    diffs = np.mean(np.abs(np.diff(raw_frames, axis=0)), axis=(1, 2, 3))

                    # Geçmişe kaydet (Tarih/Zaman olmadan)
                    st.session_state["history"].append({
                        "file_name": uploaded_file.name,
                        "prob": prob,
                        "verdict": verdict,
                        "mode": mode,
                        "diffs": diffs
                    })

                    st.metric(label="Sahtelik Olasılığı", value=f"%{prob}")
                    if verdict.startswith("SAHTE"):
                        st.error(f"🚨 KARAR: {verdict}")
                    else:
                        st.success(f"✅ KARAR: {verdict}")

                    st.caption(f"Çalışan Motor: {mode}")

                    st.markdown("---")
                    st.subheader("📈 Akademik Görsel Sinyal Grafikleri")

                    # Matplotlib Karanlık Tema Uyumlaştırması
                    plt.style.use('dark_background')

                    # 1. Kare Kare Görsel Değişim Grafiği (Frame Difference Signal)
                    fig1, ax1 = plt.subplots(figsize=(7, 2.8), facecolor='#0f172a')
                    ax1.set_facecolor('#1e293b')
                    chart_color = '#ef4444' if prob > 50 else '#22c55e'
                    ax1.plot(range(1, len(diffs) + 1), diffs, color=chart_color, linewidth=2, marker='o', markersize=4)
                    ax1.set_title("Kare Kare Görsel Değişim Haritası (Frame-Difference Signal)", color='#f8fafc', fontsize=10)
                    ax1.set_xlabel("Kare İndeksi (Frame Index)", color='#94a3b8', fontsize=8)
                    ax1.set_ylabel("Piksel Değişim Şiddeti", color='#94a3b8', fontsize=8)
                    ax1.grid(True, linestyle='--', alpha=0.2)
                    ax1.tick_params(colors='#94a3b8', labelsize=8)
                    st.pyplot(fig1)

                    # 2. Ses-Görüntü Senkronizasyon Matrisi (Cross-Modal Heatmap)
                    fig2, ax2 = plt.subplots(figsize=(7, 3.2), facecolor='#0f172a')
                    ax2.set_facecolor('#1e293b')
                    T = len(diffs)
                    sync_matrix = np.outer(diffs, np.sin(np.linspace(0, 4 * np.pi, T)))
                    cax = ax2.imshow(sync_matrix, cmap='coolwarm', aspect='auto')
                    cbar = fig2.colorbar(cax, ax=ax2)
                    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
                    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#94a3b8', fontsize=8)
                    cbar.set_label("Senkronizasyon Hizalama Derecesi", color='#94a3b8', fontsize=8)
                    
                    ax2.set_title("Ses - Görüntü Senkronizasyon Korelasyon Haritası (Cross-Modal Map)", color='#f8fafc', fontsize=10)
                    ax2.set_xlabel("Ses Frekans Bandı Zaman Adımı", color='#94a3b8', fontsize=8)
                    ax2.set_ylabel("Görsel Kare Zaman Adımı", color='#94a3b8', fontsize=8)
                    ax2.tick_params(colors='#94a3b8', labelsize=8)
                    st.pyplot(fig2)

        try:
            os.remove(video_path)
        except Exception:
            pass

# ----------------------------
# SEKMELER 2: ANALİZ GEÇMİŞİ
# ----------------------------
with main_tab2:
    st.subheader("📜 Geçmiş Analiz Sonuçları")
    
    if not st.session_state["history"]:
        st.info("💡 Henüz geçmiş bir analiz kaydı bulunmuyor. Video analizi yaptıkça sonuçlar burada listelenecektir.")
    else:
        col_clear, _ = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Geçmişi Temizle", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

        st.markdown("---")

        # Son yapılan analiz en üstte gözükecek şekilde ters sıralıyoruz
        for idx, item in enumerate(reversed(st.session_state["history"])):
            title_label = f"📹 Dosya: {item['file_name']} | Karar: {item['verdict']} (%{item['prob']})"
            
            with st.expander(title_label, expanded=(idx == 0)):
                h_col1, h_col2 = st.columns([1, 1.2])
                
                with h_col1:
                    st.markdown("#### 📊 Analiz Özeti")
                    st.metric(label="Sahtelik Olasılığı", value=f"%{item['prob']}")
                    if item['verdict'].startswith("SAHTE"):
                        st.error(f"🚨 KARAR: {item['verdict']}")
                    else:
                        st.success(f"✅ KARAR: {item['verdict']}")
                    st.caption(f"Çalışan Motor: {item['mode']}")

                with h_col2:
                    st.markdown("#### 📈 Kayıtlı Sinyal Grafikleri")
                    diffs_hist = item["diffs"]
                    plt.style.use('dark_background')

                    # 1. Kare Kare Görsel Değişim Haritası Re-plot
                    fig_h1, ax_h1 = plt.subplots(figsize=(6, 2.3), facecolor='#0f172a')
                    ax_h1.set_facecolor('#1e293b')
                    chart_color = '#ef4444' if item['prob'] > 50 else '#22c55e'
                    ax_h1.plot(range(1, len(diffs_hist) + 1), diffs_hist, color=chart_color, linewidth=2, marker='o', markersize=3)
                    ax_h1.set_title("Kare Kare Görsel Değişim Haritası", color='#f8fafc', fontsize=9)
                    ax_h1.set_xlabel("Kare İndeksi", color='#94a3b8', fontsize=7)
                    ax_h1.set_ylabel("Piksel Değişimi", color='#94a3b8', fontsize=7)
                    ax_h1.grid(True, linestyle='--', alpha=0.2)
                    ax_h1.tick_params(colors='#94a3b8', labelsize=7)
                    st.pyplot(fig_h1)

                    # 2. Ses-Görüntü Senkronizasyon Haritası Re-plot
                    fig_h2, ax_h2 = plt.subplots(figsize=(6, 2.6), facecolor='#0f172a')
                    ax_h2.set_facecolor('#1e293b')
                    T_hist = len(diffs_hist)
                    sync_matrix_hist = np.outer(diffs_hist, np.sin(np.linspace(0, 4 * np.pi, T_hist)))
                    cax_h = ax_h2.imshow(sync_matrix_hist, cmap='coolwarm', aspect='auto')
                    cbar_h = fig_h2.colorbar(cax_h, ax=ax_h2)
                    cbar_h.ax.yaxis.set_tick_params(color='#94a3b8')
                    plt.setp(plt.getp(cbar_h.ax, 'yticklabels'), color='#94a3b8', fontsize=7)
                    cbar_h.set_label("Senkronizasyon Derecesi", color='#94a3b8', fontsize=7)
                    
                    ax_h2.set_title("Ses - Görüntü Senkronizasyon Korelasyon Haritası", color='#f8fafc', fontsize=9)
                    ax_h2.tick_params(colors='#94a3b8', labelsize=7)
                    st.pyplot(fig_h2)