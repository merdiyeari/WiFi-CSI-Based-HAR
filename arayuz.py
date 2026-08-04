import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

# Sayfa ayarları
st.set_page_config(page_title="WiFi Aktivite Tanıma", layout="wide")

# 1. Dosyaları Yükle
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model_wifi.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('pca.pkl', 'rb') as f: pca = pickle.load(f)
    return model, scaler, pca

try:
    model, scaler, pca = load_assets()
except:
    st.error("Model dosyaları (h5, pkl) bulunamadı! Lütfen aynı klasörde olduklarından emin olun.")

# Hareket isimleri ve emojiler
hareket_isimleri = {
    0: 'Yatma 🛌', 1: 'Düşme ⚠️', 2: 'Yürüme 🚶', 
    3: 'Bir Şey Alma 📦', 4: 'Koşma 🏃', 5: 'Oturma 🪑', 6: 'Kalkma ⬆️'
}

st.title("📡 WiFi CSI Aktivite Tanıma Sistemi")

# Sidebar - Dosya Yükleme ve Filtreleme
st.sidebar.header("📂 Veri Yükleme")
data_file = st.sidebar.file_uploader("X_test verisi (.npy)", type=['npy'])
label_file = st.sidebar.file_uploader("y_test etiketleri (.npy)", type=['npy'])

if data_file and label_file:
    X_test = np.load(data_file)
    y_test = np.load(label_file)

    st.sidebar.divider()
    st.sidebar.header("🎯 Hareket Seçimi")
    
    # Hangi hareketi görmek istiyorsak onu seçelim
    secenekler = ["Hepsi"] + list(hareket_isimleri.values())
    filtre = st.sidebar.selectbox("Harekete Göre Filtrele:", secenekler)

    # Filtreye göre indeksleri bul
    if filtre == "Hepsi":
        indeksler = np.arange(len(y_test))
    else:
        hedef_id = [k for k, v in hareket_isimleri.items() if v == filtre][0]
        indeksler = np.where(y_test == hedef_id)[0]

    if len(indeksler) > 0:
        secilen_siradaki = st.sidebar.select_slider(f"{filtre} Örnekleri", options=range(len(indeksler)))
        index = indeksler[secilen_siradaki]

        # 2. Veri İşleme ve Tahmin[cite: 1]
        tek_veri = X_test[index]
        
        # Ön işleme adımları (Sıralama çok önemli!)[cite: 1]
        reshaped = tek_veri.reshape(-1, 90)
        scaled = scaler.transform(reshaped)
        pca_data = pca.transform(scaled)
        model_input = pca_data.reshape(1, 250, 10)

        # Tahmin yap ve olasılıkları al
        tahmin_olasiliklari = model.predict(model_input, verbose=0)[0]
        tahmin_id = np.argmax(tahmin_olasiliklari)
        guven_yuzdesi = tahmin_olasiliklari[tahmin_id] * 100
        gercek_id = y_test[index]

        # 3. Görselleştirme
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📊 Örnek No: {index} - WiFi Sinyal Deseni")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(tek_veri[:, 0], color='#1f77b4') # Subcarrier 0 çizimi[cite: 1]
            ax.set_title(f"Ham Sinyal (Gerçek: {hareket_isimleri[gercek_id]})")
            st.pyplot(fig)

        with col2:
            st.subheader("🤖 Yapay Zeka Analizi")
            
            # Sonuç Kutusu
            st.info(f"**Tahmin Edilen:** {hareket_isimleri[tahmin_id]}")
            st.write(f"**Gerçek Hareket:** {hareket_isimleri[gercek_id]}")
            
            # Doğruluk Kontrolü
            if tahmin_id == gercek_id:
                st.success(f"✅ Başarılı! (Güven: %{guven_yuzdesi:.2f})")
            else:
                st.error(f"❌ Hatalı! (Güven: %{guven_yuzdesi:.2f})")

            # Yüzdesel Dağılım Çizelgesi
            st.write("#### Olasılık Dağılımı")
            for i, olasilik in enumerate(tahmin_olasiliklari):
                st.write(f"{hareket_isimleri[i]}:")
                st.progress(float(olasilik))
    else:
        st.sidebar.warning("Bu kategoriye ait veri bulunamadı.")
else:
    st.info("Lütfen sol menüden .npy dosyalarını yükleyerek başlayın.")