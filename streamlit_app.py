import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageOps, ImageEnhance
import joblib
import time
import os
import gdown


# --- Setup Streamlit Page ---
st.set_page_config(
    page_title="Sifferklassificering",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initiera Session State ---
for key in ["prediction", "processed", "canvas_key"]:
    st.session_state.setdefault(key, None)

# --- Modellhantering ---
MODEL_PATH = "best_model.pkl"
FILE_ID = "17l9LqfvC0q8tRs37XnhPPNeAHaf8iCrU"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔽 Laddar ner modellen"):
            try:
                gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=True, fuzzy=True, use_cookies=False)
            except Exception as e:
                st.error(f"❌ Nedladdning misslyckades: {e}")
                raise FileNotFoundError("Modellen kunde inte laddas ner.")

            if not os.path.exists(MODEL_PATH):
                st.error("❌ Filen laddades inte ner korrekt.")
                raise FileNotFoundError("Filen finns inte efter nedladdning.")

    return joblib.load(MODEL_PATH)

model = load_model()

# --- Preprocessing ---
def preprocess_canvas(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1)

    img_array = np.array(img)
    threshold = 9.5
    img_array = np.where(img_array < threshold, 0, 255).astype("uint8")

    return img_array.reshape(1, 784)

# --- Sidebar ---
st.sidebar.markdown("### 📌 Om appen")
st.sidebar.info("""
En Streamlit applikation som klassificerar handskrivna siffror i realtid med hjälp av maskininlärning.

Projektet kombinerar modellträning, utvärdering, utveckling och ett interaktivt gränssnitt som visualiserar modellens gissning och sannolikhetsfördelning.

Arbetet är en del av min utbildning, Data Scientist - EC Utbildning, och syftar till att demonstrera hur maskininlärning kan användas i praktiken.
""")

st.sidebar.markdown("---")
stroke_width = st.sidebar.slider("✏️ Pennstorlek", 1, 10, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Kontakt")
st.sidebar.markdown("""
<a href="https://www.linkedin.com/in/lence-majzovska-9837702a7/" target="_blank">📎 LinkedIn</a>
<a href="https://github.com/lencemajzovska" target="_blank">💻 GitHub</a>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
    <div style='font-weight: bold;'>Lence Majzovska</div>
    <div style='color: gray;'>Made with ❤️ using Python & Streamlit</div>
</div>
""", unsafe_allow_html=True)

# --- Rubrik ---
st.title("🔢 Klassificering av siffror")

with st.expander("🔍 Så använder du appen"):
    st.markdown("""
    - ✅ **Rita** en siffra (0–9) i rutan.
    - 🔍 **Modellen analyserar** ritningen.
    - 🎯 **Resultatet visas** direkt.
    - 📊 **Sannolikheter** visas grafiskt.
    """)

# --- Layout (Kolumner) ---
col1, col2, col3 = st.columns([2, 2, 4])

# --- Kolumn 1: Canvas ---
with col1:
    st.markdown("<h4 style='text-align: left;'>🎨 Rita din siffra:</h4>", unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key or "canvas",
        display_toolbar=False,
    )

    col_save, col_clear, _ = st.columns([20, 20, 6])
    with col_save:
         if st.button("💾 Spara", use_container_width=True):
             st.toast("Spara är inte tillgängligt just nu", icon="ℹ️")

    with col_clear:
         if st.button("🗑️ Rensa", use_container_width=True):
             st.session_state.canvas_key = f"canvas_{time.time()}"
             st.session_state.prediction = None
             st.session_state.processed = None
             st.rerun()

# --- Bearbeta & Prediktera ---
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data[:, :, 0].astype("uint8"))
    processed = preprocess_canvas(img)

    if processed.sum() > 1500:
        prediction = int(model.predict(processed)[0])
        st.session_state.prediction = prediction
        st.session_state.processed = processed
    else:
        st.session_state.prediction = None
        st.session_state.processed = None

# --- Kolumn 2: Prediktion ---
with col2:
    st.markdown("<h4 style='text-align: left;'>🤖 Modellens gissning:</h4>", unsafe_allow_html=True)

    prediction = st.session_state.prediction
    if prediction is not None:
        st.markdown(f"""
            <div style="width:280px;height:280px;display:flex;align-items:center;justify-content:center;background-color:#262730;color:white;font-size:100px;font-weight:bold;box-shadow:0 0 10px rgba(0,0,0,0.4);">{prediction}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="width:280px;height:280px;background-color:#262730;box-shadow:0 0 10px rgba(0,0,0,0.4);"></div>
        """, unsafe_allow_html=True)

# --- Kolumn 3: Sannolikheter ---
with col3:
    st.markdown("<h4 style='text-align: center;'>🤔 Modellens säkerhet:</h4>", unsafe_allow_html=True)

    probs = model.predict_proba(st.session_state.processed)[0] if st.session_state.processed is not None else np.zeros(10)
    df_probs = pd.DataFrame({"Siffra": range(10), "Sannolikhet": probs})

    chart = alt.Chart(df_probs).mark_bar().encode(
        x=alt.X("Siffra:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Sannolikhet:Q", axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1])),
        tooltip=["Siffra", alt.Tooltip("Sannolikhet", format=".2%")]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

# --- Tipsruta ---
st.info("💡 Tips: För bästa resultat – rita långsamt och tydligt.")