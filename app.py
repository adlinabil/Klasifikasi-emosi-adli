import os
import cv2
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard CNN Klasifikasi Emosi Wajah",
    page_icon="😊",
    layout="wide"
)

DATASET_PATH = "D:\Citra Emosi Wajah Menggunakan CNN\Citra Emosi Wajah Menggunakan CNN\app.py"
MODEL_PATH = "model_emotion_cnn.h5"
ARTIFACTS_PATH = "evaluation_artifacts.pkl"

# ==========================================
# FUNGSI PEMUATAN DATA & MODEL
# ==========================================
@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

@st.cache_data
def load_artifacts():
    if os.path.exists(ARTIFACTS_PATH):
        with open(ARTIFACTS_PATH, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def scan_local_dataset(path):
    if not os.path.exists(path):
        return {}
    classes = sorted([
        c for c in os.listdir(path) 
        if os.path.isdir(os.path.join(path, c)) and not c.startswith('.')
    ])
    counts = {}
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    for cls in classes:
        cls_folder = os.path.join(path, cls)
        counts[cls] = len([f for f in os.listdir(cls_folder) if f.lower().endswith(valid_ext)])
    return counts

def compute_gradcam(img_array, model, last_conv_layer_name='last_conv_layer'):
    try:
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

# ==========================================
# HEADER & DESKRIPSI PROYEK
# ==========================================
st.title("😊 Klasifikasi Citra Emosi Wajah Menggunakan CNN")
st.markdown("""
**Dashboard Interaktif — Analisis Big Data (Convolutional Neural Network)**  
Menampilkan ringkasan dataset citra emosi wajah, distribusi kelas, kurva konvergensi pelatihan (*Accuracy* & *Loss*), *Confusion Matrix*, metrik evaluasi (*Accuracy, Precision, Recall, F1-Score*), serta fitur uji prediksi citra baru secara *real-time*.
""")
st.divider()

model = load_cnn_model()
artifacts = load_artifacts()
scanned_counts = scan_local_dataset(DATASET_PATH)

if artifacts is not None:
    classes = artifacts['classes']
    class_counts = artifacts['class_counts']
    total_images = sum(class_counts.values())
    train_count = artifacts['train_samples']
    val_count = artifacts['val_samples']
elif scanned_counts:
    classes = list(scanned_counts.keys())
    class_counts = scanned_counts
    total_images = sum(class_counts.values())
    train_count = int(total_images * 0.8)
    val_count = total_images - train_count
else:
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    class_counts = {c: 0 for c in classes}
    total_images, train_count, val_count = 0, 0, 0

# ==========================================
# SIDEBAR NAVIGASI
# ==========================================
st.sidebar.header("🧭 Navigasi Dashboard")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "📊 Ringkasan & Distribusi Dataset",
        "📈 Hasil Pelatihan & Evaluasi Model",
        "🔮 Uji Prediksi Gambar (Real-Time)"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Path Dataset Lokal:**")
st.sidebar.code(DATASET_PATH, language="bash")
st.sidebar.markdown(f"**Status Model CNN:** {'✅ Aktif' if model else '⚠️ Belum Dilatih'}")

# ==========================================
# HALAMAN 1: RINGKASAN & DISTRIBUSI DATASET
# ==========================================
if menu == "📊 Ringkasan & Distribusi Dataset":
    st.subheader("1. Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Kelas Emosi", f"{len(classes)} Kelas")
    col2.metric("Total Gambar", f"{total_images:,} Citra")
    col3.metric("Data Latih (80%)", f"{train_count:,} Citra")
    col4.metric("Data Uji / Validasi (20%)", f"{val_count:,} Citra")

    st.subheader("2. Grafik Distribusi Data per Kelas")
    if total_images > 0:
        df_dist = pd.DataFrame(list(class_counts.items()), columns=['Kelas Emosi', 'Jumlah Gambar'])
        df_dist = df_dist.sort_values(by='Jumlah Gambar', ascending=False).reset_index(drop=True)

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=df_dist, x='Kelas Emosi', y='Jumlah Gambar', hue='Kelas Emosi', palette='viridis', legend=False, ax=ax)
            ax.set_title("Distribusi Jumlah Citra pada Setiap Kelas Emosi", fontweight='bold')
            for i, v in enumerate(df_dist['Jumlah Gambar']):
                ax.text(i, v + (max(df_dist['Jumlah Gambar']) * 0.01), str(v), ha='center', fontweight='bold')
            st.pyplot(fig)

        with col_table:
            df_dist['Proporsi (%)'] = (df_dist['Jumlah Gambar'] / total_images * 100).round(2)
            st.dataframe(df_dist, use_container_width=True, hide_index=True)

    if os.path.exists(DATASET_PATH):
        st.subheader("3. Sampel Visual Masing-Masing Kelas Emosi")
        cols = st.columns(len(classes))
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        for idx, cls in enumerate(classes):
            cls_folder = os.path.join(DATASET_PATH, cls)
            if os.path.exists(cls_folder):
                imgs = [f for f in os.listdir(cls_folder) if f.lower().endswith(valid_ext)]
                if imgs:
                    img_path = os.path.join(cls_folder, imgs[0])
                    cols[idx].image(Image.open(img_path), caption=f"Emosi: {cls}", use_container_width=True)

# ==========================================
# HALAMAN 2: HASIL PELATIHAN & EVALUASI MODEL
# ==========================================
elif menu == "📈 Hasil Pelatihan & Evaluasi Model":
    if artifacts is None:
        st.warning("⚠️ File `evaluation_artifacts.pkl` belum tersedia. Jalankan notebook `Klasifikasi_Emosi_CNN.ipynb` terlebih dahulu.")
    else:
        st.subheader("1. Hasil Evaluasi Model CNN (Accuracy, Precision, Recall, F1-Score)")
        m = artifacts['metrics']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (Akurasi)", f"{m['accuracy']*100:.2f}%")
        c2.metric("Precision (Presisi)", f"{m['precision']*100:.2f}%")
        c3.metric("Recall (Sensitivitas)", f"{m['recall']*100:.2f}%")
        c4.metric("F1-Score", f"{m['f1_score']*100:.2f}%")

        st.divider()
        st.subheader("2. Grafik Hasil Pelatihan (Akurasi & Loss)")
        hist = artifacts['history']
        fig_hist, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        axes[0].plot(hist['accuracy'], label='Train Accuracy', marker='o')
        axes[0].plot(hist['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[0].set_title('Grafik Akurasi Pelatihan vs Validasi', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].plot(hist['loss'], label='Train Loss', marker='o', color='crimson')
        axes[1].plot(hist['val_loss'], label='Validation Loss', marker='s', color='orange')
        axes[1].set_title('Grafik Loss Pelatihan vs Validasi', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig_hist)

        st.divider()
        col_cm, col_rep = st.columns([1.1, 1])

        with col_cm:
            st.subheader("3. Confusion Matrix")
            cm = artifacts['confusion_matrix']
            fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax_cm)
            ax_cm.set_xlabel('Label Prediksi (Predicted)')
            ax_cm.set_ylabel('Label Asli (Actual)')
            ax_cm.set_title('Confusion Matrix Klasifikasi Emosi', fontweight='bold')
            st.pyplot(fig_cm)

        with col_rep:
            st.subheader("4. Ringkasan Evaluasi per Kelas")
            df_rep = artifacts['classification_report']
            st.dataframe(df_rep.style.format("{:.4f}"), use_container_width=True)

# ==========================================
# HALAMAN 3: UJI PREDIKSI GAMBAR BARU (REAL-TIME)
# ==========================================
elif menu == "🔮 Uji Prediksi Gambar (Real-Time)":
    st.subheader("Uji Prediksi Emosi Wajah Secara Real-Time")
    st.write("Unggah gambar wajah untuk melihat hasil prediksi emosi secara langsung beserta area fokus fitur wajah (*Feature Importance*).")

    uploaded_file = st.file_uploader("Unggah file gambar wajah (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if model is None:
            st.error("❌ Model `model_emotion_cnn.h5` belum ditemukan. Jalankan notebook terlebih dahulu.")
        else:
            image_pil = Image.open(uploaded_file)
            img_gray = image_pil.convert('L')
            img_resized = img_gray.resize((48, 48))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_input = np.expand_dims(img_array, axis=(0, -1))

            preds = model.predict(img_input, verbose=0)[0]
            pred_idx = np.argmax(preds)
            pred_emotion = classes[pred_idx]
            confidence = preds[pred_idx] * 100

            col_img1, col_img2, col_res = st.columns([1, 1, 1.5])

            with col_img1:
                st.markdown("**Citra Wajah Asli:**")
                st.image(image_pil, caption="Gambar Diunggah", use_container_width=True)

            with col_img2:
                st.markdown("**Feature Importance (Grad-CAM):**")
                heatmap = compute_gradcam(img_input, model, 'last_conv_layer')
                if heatmap is not None:
                    heatmap_resized = cv2.resize(heatmap, (48, 48))
                    fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
                    ax_cam.imshow(img_array, cmap='gray')
                    ax_cam.imshow(heatmap_resized, cmap='jet', alpha=0.5)
                    ax_cam.axis('off')
                    st.pyplot(fig_cam)
                else:
                    st.image(img_resized, caption="Citra 48x48 Grayscale", use_container_width=True)

            with col_res:
                st.markdown("### 🎯 Hasil Prediksi Real-Time:")
                st.success(f"**Emosi Terdeteksi: {pred_emotion.upper()}** (Confidence: **{confidence:.2f}%**)")

                df_probs = pd.DataFrame({
                    'Kelas Emosi': classes,
                    'Probabilitas (%)': (preds * 100).round(2)
                }).sort_values(by='Probabilitas (%)', ascending=True)

                fig_prob, ax_prob = plt.subplots(figsize=(6, 3.5))
                ax_prob.barh(df_probs['Kelas Emosi'], df_probs['Probabilitas (%)'], color='teal')
                ax_prob.set_xlabel('Probabilitas (%)')
                ax_prob.set_xlim(0, 100)
                ax_prob.set_title('Probabilitas Prediksi Setiap Kelas Emosi', fontweight='bold')
                st.pyplot(fig_prob)