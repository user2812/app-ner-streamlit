import streamlit as st
from transformers import pipeline
import pandas as pd
import fitz  # PyMuPDF pour lire les PDF

# 🔁 Chargement des deux modèles
@st.cache_resource
def load_pipeline(model_name):
    return pipeline("ner", model=model_name, aggregation_strategy="simple")

ner_camembert = load_pipeline("Jean-Baptiste/camembert-ner")
ner_mbert = load_pipeline("Davlan/bert-base-multilingual-cased-ner-hrl")

# 🎨 Couleurs HTML pour chaque type d'entité
COLORS = {"PER": "#ffe599", "LOC": "#b6d7a8", "ORG": "#9fc5e8", "MISC": "#f9cb9c"}

# 📌 Mise en surbrillance HTML
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

# 🌙 Style personnalisé (mode sombre)
st.markdown("""
    <style>
    body { background-color: #1e1e1e; color: white; }
    textarea, .stTextInput > div > div > input {
        background-color: #2e2e2e !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Interface
st.title("🧠 NER interactif : CamemBERT vs mBERT")
st.markdown("Analyse de texte pour détecter les entités nommées (personnes, lieux, organisations...).")

# 📂 Upload de fichier
uploaded_file = st.file_uploader("📂 Charger un fichier .txt ou .pdf", type=["txt", "pdf"])
texte = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        texte = " ".join(page.get_text() for page in doc)
    else:
        texte = uploaded_file.read().decode("utf-8")
else:
    texte = st.text_area("✍️ Ou saisir un texte :", height=200, value="Emmanuel Macron a rencontré le président de l’ONU à Paris.")

# 🕓 Historique des textes analysés
if "history" not in st.session_state:
    st.session_state.history = []

# ▶️ Analyse du texte
if st.button("Analyser avec CamemBERT et mBERT") and texte.strip():
    with st.spinner("🔍 Analyse en cours..."):
        ent1 = ner_camembert(texte)
        ent2 = ner_mbert(texte)

        # Sauvegarde dans l'historique
        st.session_state.history.append({
            "texte": texte,
            "camembert": ent1,
            "mbert": ent2
        })

        # 🔎 Visualisation annotée
        st.subheader("📝 Texte annoté (CamemBERT)")
        st.markdown(highlight_entities(texte, ent1), unsafe_allow_html=True)

        st.subheader("📝 Texte annoté (mBERT)")
        st.markdown(highlight_entities(texte, ent2), unsafe_allow_html=True)

        # 📊 Résultats tabulaires
        df1 = pd.DataFrame(ent1)[["word", "entity_group", "score"]].rename(columns={
            "word": "Texte", "entity_group": "Type d'entité", "score": "Confiance"})
        df2 = pd.DataFrame(ent2)[["word", "entity_group", "score"]].rename(columns={
            "word": "Texte", "entity_group": "Type d'entité", "score": "Confiance"})

        st.markdown("### 📊 Résultats CamemBERT")
        st.dataframe(df1)

        st.markdown("### 📊 Résultats mBERT")
        st.dataframe(df2)

        # 📥 Télécharger les résultats combinés
        df1["Modèle"] = "CamemBERT"
        df2["Modèle"] = "mBERT"
        df_all = pd.concat([df1, df2], ignore_index=True)
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Télécharger les résultats (CSV)", csv, "entites_comparées.csv", "text/csv")

# 🕓 Historique affiché en bas
if st.session_state.history:
    with st.expander("🕓 Voir l'historique des analyses de cette session"):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Texte #{i}** : `{item['texte'][:100]}...`")
            st.markdown("Entités CamemBERT : " + ", ".join([e['word'] for e in item["camembert"]]))
            st.markdown("Entités mBERT : " + ", ".join([e['word'] for e in item["mbert"]]))
            st.markdown("---")
