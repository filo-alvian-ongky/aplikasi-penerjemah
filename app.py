# app.py
import streamlit as st
from transformers import pipeline
import torch

# Judul dan deskripsi Aplikasi
st.set_page_config(page_title="Penerjemah Multi-Bahasa", page_icon="🌐", layout="centered")
st.title("🌐 Aplikasi Terjemahan Multi-Bahasa")
st.write("Ditenagai oleh Model NLLB dari Meta AI. Dibuat untuk Semarang.")

# --- Fungsi untuk Memuat Model dengan Cache ---
# @st.cache_resource memberitahu Streamlit untuk menjalankan fungsi ini sekali saja.
@st.cache_resource
def load_translator_pipeline():
    """
    Memuat model NLLB dari Hugging Face Hub.
    Fungsi ini hanya akan berjalan sekali saat aplikasi pertama kali di-boot.
    """
    # Menampilkan pesan di aplikasi dan di log server
    st.info("Memuat model penerjemah... Proses ini mungkin memakan waktu beberapa menit saat booting pertama.")
    print("Memulai proses pemuatan model NLLB...")
    
    model_name = "facebook/nllb-200-distilled-600M"
    
    translator = pipeline(
        'translation',
        model=model_name,
        # Menggunakan CPU secara eksplisit karena resource GPU di Streamlit Cloud terbatas
        device=-1 
    )
    
    print("Model NLLB berhasil dimuat.")
    # Menghapus pesan info setelah model dimuat
    st.empty() 
    return translator

# Panggil fungsi untuk memuat model (akan menggunakan cache jika sudah ada)
translator = load_translator_pipeline()

# --- Antarmuka Pengguna (UI) ---
st.header("Pilih Arah Terjemahan")

option = st.selectbox(
    'Pilih tugas terjemahan:',
    ('Inggris ➡️ Korea', 'Korea ➡️ Inggris', 'Korea ➡️ Indonesia', 'Indonesia ➡️ Korea'),
    label_visibility="collapsed"
)

st.header("Masukkan Teks")
input_text = st.text_area("Ketik teks yang ingin Anda terjemahkan:", height=150, placeholder="Contoh: Hello, how are you?")

if st.button("Terjemahkan Sekarang 🚀", type="primary"):
    if input_text:
        with st.spinner("Menerjemahkan... Mohon tunggu sebentar."):
            try:
                # Logika pemilihan bahasa
                lang_map = {
                    'Inggris ➡️ Korea': ('eng_Latn', 'kor_Hang'),
                    'Korea ➡️ Inggris': ('kor_Hang', 'eng_Latn'),
                    'Korea ➡️ Indonesia': ('kor_Hang', 'ind_Latn'),
                    'Indonesia ➡️ Korea': ('ind_Latn', 'kor_Hang')
                }
                src_lang, tgt_lang = lang_map[option]
                
                result = translator(input_text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=128)
                
                # Tampilkan hasil
                st.subheader("🎉 Hasil Terjemahan:")
                st.success(result[0]['translation_text'])

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Mohon masukkan teks untuk diterjemahkan.")

st.markdown("---")
st.write(f"Waktu saat ini di Semarang: {pd.Timestamp.now(tz='Asia/Jakarta').strftime('%A, %d %B %Y, %H:%M:%S WIB')}")