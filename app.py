import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# 1. Configuration de la page
st.set_page_config(page_title="Scanner Médical IA", page_icon="🩻", layout="centered")

# 2. Chargement sécurisé du modèle TensorFlow/Keras
@st.cache_resource
def charger_modele_medical():
    # compile=False évite les erreurs AttributeError sur le Cloud
    return tf.keras.models.load_model('ton_modele_scanner.keras', compile=False)

# Si ton fichier de modèle a un autre nom (ex: 'model.h5'), change-le ici :
model = charger_modele_medical()

# Liste des pathologies (⚠️ À modifier selon les classes de ton modèle)
classes_medicales = ['Normal', 'Pneumonie', 'Tumeur Détectée', 'Fracture']

# 3. Interface Utilisateur
st.title("🩻 Analyse de Scanners Médicaux par IA")
st.write("Cette application utilise un modèle de Deep Learning pour analyser les images médicales et l'IA Gemini pour générer un pré-rapport.")

# Barre latérale pour configurer la clé API Gemini de manière sécurisée (Règle le problème de la ligne 44)
st.sidebar.header("🔑 Configuration")
api_key = st.sidebar.text_input("Entrez votre clé API Google Gemini :", type="password")

# Zone d'upload du scanner
fichier_upload = st.file_uploader("Téléchargez un scanner ou une radiographie (png, jpg, jpeg)...", type=["jpg", "jpeg", "png"])

# 4. Traitement de l'image et analyse
if fichier_upload is not None:
    image = Image.open(fichier_upload)
    st.image(image, caption="Scanner téléchargé", use_container_width=True)
    
    # --- PARTIE 1 : Prédiction du modèle TensorFlow ---
    with st.spinner("Analyse du scanner par le modèle d'imagerie..."):
        # Redimensionnement (⚠️ Change 180, 180 par la taille d'entraînement de ton modèle)
        image_redimensionnee = image.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(image_redimensionnee)
        img_array = tf.expand_dims(img_array, 0)
        
        # Prédiction
        predictions = model.predict(img_array)
        index_prediction = np.argmax(predictions[0])
        pathologie_detectee = classes_medicales[index_prediction]
        confiance = 100 * np.max(tf.nn.softmax(predictions[0]))

    # Affichage du résultat du modèle
    st.success(f"🔬 Analyse d'imagerie : **{pathologie_detectee}**")
    st.info(f"📈 Indice de certitude : **{confiance:.2f}%**")
    
    # --- PARTIE 2 : Génération du rapport par Gemini ---
    if api_key:
        with st.spinner("Génération du pré-rapport médical par Gemini..."):
            try:
                # Initialisation de Gemini avec la clé entrée par l'utilisateur
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"Agis en tant qu'expert radiologue. Rédige un court compte-rendu médical fictif basé sur une analyse d'imagerie qui a détecté : '{pathologie_detectee}' avec une confiance de {confiance:.2f}%. Ajoute une section de recommandations."
                
                response = gemini_model.generate_content(prompt)
                
                st.subheader("📝 Pré-rapport Médical Automatique")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Erreur lors de la génération du rapport : {e}")
    else:
        st.warning("💡 Conseil : Entrez votre clé API Gemini dans la barre latérale pour obtenir un compte-rendu textuel automatique.")