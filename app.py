import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page
st.set_page_config(page_title="Scanner Médical IA", page_icon="🩻", layout="centered")

st.title("🩻 Analyse de Scanners Médicaux par IA")
st.write("Téléchargez une image de scanner ou de radiographie pour obtenir un diagnostic automatique instantané.")

# 2. Chargement du modèle Keras
@st.cache_resource
def charger_modele_medical():
    return tf.keras.models.load_model('modele_medical.keras', compile=False)

model = charger_modele_medical()

# Liste des pathologies (L'ordre doit correspondre à celui de ton entraînement)
classes_medicales = ['Normal', 'Pneumonie', 'Tumeur Détectée', 'Fracture']

# 3. Zone de téléchargement du scanner
fichier_upload = st.file_uploader("Sélectionnez une image médicale (png, jpg, jpeg)...", type=["jpg", "jpeg", "png"])

# 4. Traitement mathématique et prédiction
if fichier_upload is not None:
    image = Image.open(fichier_upload)
    st.image(image, caption="Scanner à analyser", use_container_width=True)
    
    with st.spinner("Analyse des structures de l'image..."):
        # Redimensionnement standard (150x150)
        image_redimensionnee = image.resize((150, 150))
        
        # Conversion en tableau de pixels
        img_array = tf.keras.utils.img_to_array(image_redimensionnee)
        
        # 🧪 --- JEU DE TESTS DE NORMALISATION ---
        # Déplace le symbole '#' pour activer la ligne correspondant à ton entraînement :
        
        # OPTION A (Pixels bruts entre 0 et 255) :
        # Pas de modification, laisse comme ça.
        
        # OPTION B (Pixels normalisés entre 0 et 1) -> ACTIVE PAR DÉFAUT :
        img_array = img_array / 255.0
        
        # OPTION C (Pixels centrés entre -1 et 1) :
        # img_array = (img_array / 127.5) - 1.0
        
        # ----------------------------------------
        
        # Ajout de la dimension Batch
        img_array = tf.expand_dims(img_array, 0)
        
        # Calcul des prédictions
        predictions_brutes = model.predict(img_array)
        
        # Utilisation de la fonction Argmax directe sur les sorties du modèle
        index_gagnant = np.argmax(predictions_brutes[0])
        pathologie_detectee = classes_medicales[index_gagnant]
        
        # Calcul des probabilités pour le tableau de bord
        scores_probabilites = tf.nn.softmax(predictions_brutes[0]).numpy()
        confiance = scores_probabilites[index_gagnant] * 100

    # --- AFFICHAGE DES RÉSULTATS ---
    st.markdown("---")
    st.subheader("🔬 Résultats de l'analyse automatique")
    
    st.success(f"Diagnostic détecté : **{pathologie_detectee}**")
    st.info(f"Indice de certitude globale : **{confiance:.2f}%**")
    
    # 📊 Tableau dynamique pour voir les pourcentages bouger
    st.subheader("📊 Répartition des probabilités")
    for i, classe in enumerate(classes_medicales):
        valeur_pourcentage = scores_probabilites[i] * 100
        st.write(f"- **{classe}** : `{valeur_pourcentage:.2f}%`")
        st.progress(float(scores_probabilites[i]))