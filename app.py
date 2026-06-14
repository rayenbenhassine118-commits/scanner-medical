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

# 3. Zone de téléchargement du scanner
fichier_upload = st.file_uploader("Sélectionnez une image médicale (png, jpg, jpeg)...", type=["jpg", "jpeg", "png"])

# 4. Traitement mathématique et prédiction
if fichier_upload is not None:
    image = Image.open(fichier_upload)
    st.image(image, caption="Scanner à analyser", use_container_width=True)
    
    with st.spinner("Analyse des structures de l'image..."):
        # Redimensionnement standard (150x150)
        image_redimensionnee = image.resize((150, 150))
        
        # Conversion en tableau de pixels et normalisation (Option B)
        img_array = tf.keras.utils.img_to_array(image_redimensionnee)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, 0)
        
        # Calcul des prédictions brutes du modèle
        predictions_brutes = model.predict(img_array)
        
        # Application automatique du Softmax pour obtenir des probabilités
        scores_probabilites = tf.nn.softmax(predictions_brutes[0]).numpy()
        
        # Nombre exact de sorties détectées par ton modèle
        nb_classes_detectees = len(scores_probabilites)
        
        # Génération automatique des noms de classes pour éviter l'IndexError
        # (Tu pourras renommer 'Classe 0', 'Classe 1' une fois que tu verras laquelle s'active)
        classes_medicales = [f"Pathologie (Type {i})" for i in range(nb_classes_detectees)]
        
        # Si le modèle a exactement 2 classes, on peut supposer :
        if nb_classes_detectees == 2:
            classes_medicales = ['Normal (Sain)', 'Anomalie Détectée']
        # Si le modèle a exactement 4 classes (comme prévu au départ) :
        elif nb_classes_detectees == 4:
            classes_medicales = ['Normal', 'Pneumonie', 'Tumeur Détectée', 'Fracture']
            
        # Extraction de la classe dominante
        index_gagnant = np.argmax(scores_probabilites)
        pathologie_detectee = classes_medicales[index_gagnant]
        confiance = scores_probabilites[index_gagnant] * 100

    # --- AFFICHAGE DES RÉSULTATS ---
    st.markdown("---")
    st.subheader("🔬 Résultats de l'analyse automatique")
    
    st.success(f"Diagnostic détecté : **{pathologie_detectee}**")
    st.info(f"Indice de certitude globale : **{confiance:.2f}%**")
    
    # 📊 Tableau dynamique sécurisé (ne causera plus jamais de IndexError)
    st.subheader("📊 Répartition des probabilités")
    for i in range(nb_classes_detectees):
        valeur_pourcentage = scores_probabilites[i] * 100
        st.write(f"- **{classes_medicales[i]}** : `{valeur_pourcentage:.2f}%`")
        st.progress(float(scores_probabilites[i]))