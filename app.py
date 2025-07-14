import streamlit as st
from transformers import pipeline
import pandas as pd

# 🎨 Couleurs HTML par type d'entité
COLORS = {
    "PER": "#ffe599",  # jaune clair
    "LOC": "#b6d7a8",  # vert menthe
    "ORG": "#9fc5e8",  # bleu doux
    "MISC": "#f9cb9c"  # rose pêche
}

# 🚀 Charger un pipeline avec cache
@st.cache_resource
def load_pipeline(model_name):
    return pipeline("ner", model=model_name, aggregation_strategy="simple")

# 🎨 Mise en évidence HTML des entités dans le texte
def highlight_entities(text, entities):
    offset = 0
    for ent in sorted(entities, key=lambda x: x["start"]):
        start = ent["start"] + offset
        end = ent["end"] + offset
        label = ent["entity_group"]
        color = COLORS.get(label, "#dddddd")
        span = f'<span style="background-color:{color}; padding:2px; border-radius:4px;">{text[start:end]}<sub> {label}</sub></span>'
        text = text[:start] + span + text[end:]
        offset += len(span) - (end - start)
    return text

# 🌑 Mode sombre (CSS injecté)
st.markdown("""
    <style>
    body { background-color: #1e1e1e; color: white; }
    textarea, .stTextInput > div > div > input {
        background-color: #2e2e2e !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Interface principale
st.title("🧠 NER interactif en français")
st.markdown("Détecte automatiquement les **entités nommées** comme les **personnes**, **lieux**, **organisations**, etc.")

# 🧪 Sélecteur de modèle
model_choice = st.selectbox("Modèle à utiliser :", [
    "Jean-Baptiste/camembert-ner", 
    "Davlan/bert-base-multilingual-cased-ner-hrl"
], format_func=lambda x: "CamemBERT (français)" if "camembert" in x else "mBERT (multilingue)")

ner = load_pipeline(model_choice)

# 📝 Zone de texte
text = st.text_area("Texte à analyser", height=200, value="Emmanuel Macron a rencontré le président de l’ONU à Paris en 2023.")

# ▶️ Analyse du texte
if st.button("Analyser les entités") and text.strip():
    with st.spinner("Extraction en cours..."):
        entities = ner(text)
        highlighted = highlight_entities(text, entities)

        # 🖼️ Texte annoté
        st.markdown("### 📝 Texte annoté :", unsafe_allow_html=True)
        st.markdown(highlighted, unsafe_allow_html=True)

        # 📋 Affichage tabulaire
        if entities:
            df = pd.DataFrame(entities)
            df = df[["word", "entity_group", "score"]]
            df.columns = ["Texte", "Type d'entité", "Confiance"]
            st.markdown("### 📊 Entités détectées :")
            st.dataframe(df)

            # 💾 Télécharger en CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les entités (CSV)", csv, "entites.csv", "text/csv")
        else:
            st.warning("Aucune entité détectée.")
