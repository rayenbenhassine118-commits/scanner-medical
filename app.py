import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# 1. Configuration de la page
st.set_page_config(page_title="Scanner Médical IA", page_icon="🩻", layout="centered")

st.title("🩻 Analyse de Scanners Médicaux par IA")
st.write("Cette application utilise un modèle de Deep Learning pour analyser les images médicales et l'IA Gemini pour générer un pré-rapport.")

# 2. Chargement sécurisé du modèle
@st.cache_resource
def charger_modele_medical():
    return tf.keras.models.load_model('modele_medical.keras', compile=False)

model = charger_modele_medical()

# Liste des pathologies (À modifier selon les classes exactes de votre modèle)
classes_medicales = ['Normal', 'Pneumonie', 'Tumeur Détectée', 'Fracture']

# 3. Barre latérale sécurisée pour la clé API Gemini
st.sidebar.header("🔑 Configuration Sécurisée")
api_key = st.sidebar.text_input("Entrez votre clé API Google Gemini :", type="password")

# Zone de téléchargement du scanner
fichier_upload = st.file_uploader("Téléchargez un scanner ou une radiographie (png, jpg, jpeg)...", type=["jpg", "jpeg", "png"])

# 4. Traitement de l'image et analyses
if fichier_upload is not None:
    image = Image.open(fichier_upload)
    st.image(image, caption="Scanner téléchargé", use_container_width=True)
    
    # --- PARTIE 1 : Prédiction du modèle d'imagerie ---
    with st.spinner("Analyse du scanner par l'IA..."):
        # Redimensionnement à la bonne taille (150x150)
        image_redimensionnee = image.resize((150, 150))
        img_array = tf.keras.utils.img_to_array(image_redimensionnee)
        
        # 🟢 CORRECTIF DE NORMALISATION : Convertit les pixels de [0-255] à [0-1]
        # Cela évite que le modèle sature et affiche toujours la même prédiction
        img_array = img_array / 255.0
        
        # Ajout de la dimension batch (1, 150, 150, 3)
        img_array = tf.expand_dims(img_array, 0)
        
        # Exécution de la prédiction
        predictions = model.predict(img_array)
        
        # Extraction de la classe
        index_prediction = np.argmax(predictions[0])
        pathologie_detectee = classes_medicales[index_prediction]
        
        # Calcul dynamique de l'indice de confiance
        confiance = 100 * np.max(predictions[0])
        
        # Sécurité : Si le modèle ne sort pas de probabilités brutes (pas de couche Softmax finale)
        if confiance < 1.0: 
            scores_softmax = tf.nn.softmax(predictions[0])
            index_prediction = np.argmax(scores_softmax)
            pathologie_detectee = classes_medicales[index_prediction]
            confiance = 100 * np.max(scores_softmax)

    # Affichage du résultat de l'imagerie
    st.success(f"🔬 Analyse d'imagerie : **{pathologie_detectee}**")
    st.info(f"📈 Indice de certitude : **{confiance:.2f}%**")
    
    # --- PARTIE 2 : Génération du rapport médical par Gemini ---
    if api_key:
        with st.spinner("Génération du compte-rendu par Gemini..."):
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-pro')
                
                prompt = f"Agis en tant qu'expert radiologue. Rédige un court compte-rendu médical fictif basé sur une analyse d'imagerie qui a détecté : '{pathologie_detectee}' avec une confiance de {confiance:.2f}%. Ajoute des recommandations."
                
                response = gemini_model.generate_content(prompt)
                
                st.subheader("📝 Pré-rapport Médical Automatique")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Erreur lors de la génération du rapport : {e}")
    else:
        st.warning("💡 Conseil : Entrez votre clé API Gemini dans la barre latérale pour obtenir le compte-rendu textuel automatique.")