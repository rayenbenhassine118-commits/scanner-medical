import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import google.generativeai as genai

# --- 1. CONFIGURATION VISUELLE DE LA PAGE ---
st.set_page_config(
    page_title="Scanner Radiologique & Assistant IA",
    page_icon="🩻",
    layout="centered"
)

# Style CSS pour forcer le look "Écran de contrôle médical"
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    h1 { color: #58a6ff !important; font-family: 'Courier New', Courier, monospace; text-align: center; }
    .stAlert { background-color: #161b22 !important; border: 1px solid #30363d !important; }
    
    /* Style personnalisé pour la sidebar (Zone Chatbot) */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    .chat-header {
        color: #58a6ff;
        font-family: 'Courier New', Courier, monospace;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. CONFIGURATION DE L'ASSISTANT ET DE L'API DANS LA SIDEBAR 💬 ---
with st.sidebar:
    st.markdown("<div class='chat-header'>🤖 ASSISTANT MÉDICAL</div>", unsafe_allow_html=True)
    
    # Ta clé d'API Google Gemini officielle intégrée directement
    
    
    st.write("---")

    # Initialisation de l'historique du chat si vide
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour Docteur. Mon système de réflexion avancé est connecté. Comment puis-je vous aider aujourd'hui ?"}
        ]

    # Affichage des messages historiques
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Zone de saisie pour l'utilisateur
    if prompt := st.chat_input("Posez votre question médicale..."):
        # Afficher le message de l'utilisateur
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Appel à l'intelligence de Google Gemini
        with st.chat_message("assistant"):
            reponse_placeholder = st.empty()
            
            try:
                # Configuration de Gemini avec la clé
                genai.configure(api_key=api_key)
                
                # Instruction système pour forcer le modèle à agir comme un expert médical
                instructions_medecin = (
                    "Tu es un assistant virtuel en radiologie et pneumologie expert. "
                    "Tu réponds de manière professionnelle, bienveillante et scientifique. "
                    "Donne des conseils clairs (symptômes, hydratation, repos, structures de la radio) "
                    "mais rappelle TOUJOURS de consulter un vrai médecin à la fin si la question implique un diagnostic."
                )
                
                # CORRECTION ICI : "gemini-1.5-flash-latest" au lieu de "gemini-1.5-flash"
                model_gemini = genai.GenerativeModel(
                    model_name="gemini-1.5-flash-latest",
                    system_instruction=instructions_medecin
                )
                
                # Envoi de la question
                response = model_gemini.generate_content(prompt)
                brute_reponse = response.text
                
            except Exception as e:
                # Affichage transparent de l'erreur en cas de problème d'API
                brute_reponse = f"⚠️ Erreur détaillée de l'IA : {e}"

            # Effet d'écriture dynamique (lettre par lettre)
            reponse = ""
            for char in brute_reponse:
                reponse += char
                time.sleep(0.005)
                reponse_placeholder.write(reponse + "▌")
            reponse_placeholder.write(reponse)
            
        st.session_state.messages.append({"role": "assistant", "content": brute_reponse})


# --- 3. PAGE PRINCIPALE : SCANNER RADIOLOGIQUE ---
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