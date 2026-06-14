import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

# --- 1. CONFIGURATION VISUELLE DE LA PAGE ---
st.set_page_config(
    page_title="Scanner Radiologique IA",
    page_icon="🩻",
    layout="centered"
)

# Style CSS pour forcer le look "Écran de contrôle médical"
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    h1 { color: #58a6ff !important; font-family: 'Courier New', Courier, monospace; text-align: center; }
    .stAlert { background-color: #161b22 !important; border: 1px solid #30363d !important; }
    </style>
""", unsafe_allow_html=True)


# --- 2. PAGE PRINCIPALE : SCANNER RADIOLOGIQUE ---
st.markdown("<h1>🩻 CENTRE DE RADIOLOGIE IA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Système autonome d'analyse pulmonaire par Réseau de Neurones Convolutif</p>", unsafe_allow_html=True)
st.write("---")

# Chargement sécurisé du modèle IA local
@st.cache_resource
def charger_intelligence():
    return tf.keras.models.load_model("modele_medical.keras")

try:
    modele = charger_intelligence()
except Exception as e:
    st.error("⚠️ Fichier 'modele_medical.keras' introuvable dans le dossier. Vérifie son emplacement.")
    st.stop()

# Formulaire d'upload de la radio
st.subheader("📥 Dépôt de la radiographie thoracique")
image_chargee = st.file_uploader("Glissez-déposez le cliché X-Ray du patient (Format JPG ou PNG)...", type=["jpg", "png", "jpeg"])

if image_chargee is not None:
    radio = Image.open(image_chargee)
    st.image(radio, caption="Cliché radiologique chargé.", use_container_width=True)
    st.write("---")
    
    # Déclencheur du scan
    if st.button("🚀 LANCER L'ANALYSE DU CLICHÉ", use_container_width=True):
        progress_text = "🔬 Analyse de la structure pulmonaire en cours. Veuillez patienter..."
        barre_chargement = st.progress(0, text=progress_text)
        
        # Faux balayage pour l'effet technologique
        for pourcentage in range(0, 101, 20):
            time.sleep(0.3)
            barre_chargement.progress(pourcentage, text=progress_text)
        
        # Formatage de l'image pour le modèle
        taille_ia = (150, 150)
        radio_preparee = ImageOps.fit(radio, taille_ia).convert('RGB')
        
        tableau_pixel = np.asarray(radio_preparee)
        batch_image = np.expand_dims(tableau_pixel, axis=0)
        
        # Prédiction finale par ton modèle d'images
        prediction = modele.predict(batch_image)[0][0]
        barre_chargement.empty()
        
        st.subheader("📋 Rapport d'analyse automatique")
        
        if prediction >= 0.5:
            confiance = prediction * 100
            st.error(f"🚨 **ANOMALIE DÉTECTÉE :** Suspicion élevée de **PNEUMONIE**.")
            st.info(f"📈 Indice de certitude de l'algorithme : **{confiance:.2f}%**")
            st.warning("⚠️ *Note : Ce résultat est généré par une IA. Une validation par un radiologue humain est obligatoire.*")
        else:
            confiance = (1 - prediction) * 100
            st.success(f"✅ **CLICHÉ NORMAL :** Les champs pulmonaires apparaissent sains et dégagés.")
            st.info(f"📈 Indice de certitude de l'algorithme : **{confiance:.2f}%**")
else:
    st.info("💡 En attente d'un cliché radiologique pour démarrer le protocole de détection.")