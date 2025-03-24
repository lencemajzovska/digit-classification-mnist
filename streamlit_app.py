import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageOps
import joblib
import time
import os
import joblib
import gdown


# Setup
st.set_page_config(
    page_title="Sifferklassificering",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="dark"
)

# Initiera session state
for key in ["prediction", "processed", "canvas_key"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.canvas_key = st.session_state.canvas_key or "canvas"


# Ladda model
@st.cache_resource
def load_model():
    file_id = "17l9LqfvC0q8tRs37XnhPPNeAHaf8iCrU"  # ← Ditt Google Drive ID
    model_path = "best_model.pkl"

    if not os.path.exists(model_path):
        st.write("🔽 Laddar ner modellen från Google Drive...")
        try:
            # fuzzy=True hanterar "Ladda ner ändå"-skyddet
            gdown.download(id=file_id, output=model_path, quiet=False, fuzzy=True, use_cookies=False)
        except Exception as e:
            st.error(f"❌ Nedladdning misslyckades: {e}")
            raise FileNotFoundError("Modellen kunde inte laddas ner.")

        if not os.path.exists(model_path):
            st.error("❌ Filen laddades inte ner korrekt.")
            raise FileNotFoundError("Filen finns inte efter nedladdning.")

        st.success("✅ Modell nedladdad!")

    return joblib.load(model_path)

# Initiera modellen om den inte redan finns i session_state
if "model" not in st.session_state:
    st.session_state.model = load_model()


# Hämta modellen från session_state
model = st.session_state.model

# Preprocessing
def preprocess_canvas(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = img.point(lambda x: 0 if x < 50 else 255)
    img_array = np.array(img).reshape(1, 784)
    return img_array.astype("uint8")


# Sidebar
st.sidebar.markdown("### 📌 Om appen")

st.sidebar.info("""
En Streamlit applikation som klassificerar handskrivna siffror i realtid med hjälp av maskininlärning. 

Projektet kombinerar modellträning, utvärdering, utveckling och ett interaktivt gränssnitt som visualiserar modellens gissning och sannolikhets- fördelning.

Arbetet är en del av min utbildning, Data Scientist - EC Utbildning,  
och syftar till att demonstrera hur maskininlärning kan användas i praktiken.
""")

st.sidebar.markdown("---")

stroke_width = st.sidebar.slider("### ✏️ Pennstorlek", 1, 25, 12)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Kontakt")

st.sidebar.markdown("""
<a href="https://www.linkedin.com/in/lence-majzovska-9837702a7/" target="_blank">📎 LinkedIn</a>  
<a href="https://github.com/lencemajzovska" target="_blank">💻 GitHub</a>  
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='text-align: center; padding-top: 30px; padding-bottom: 10px;'>
    <div style='font-size: 14px; font-weight: bold;'>Lence Majzovska</div>
    <div style='font-size: 12px; color: gray;'>  Made with ❤️ using Python & Streamlit</div>
</div>
""", unsafe_allow_html=True)

# Rubrik
st.title("🔢 Klassificering av siffror")
with st.expander("🔍 Så använder du appen"):
    st.markdown("""
    - ✅ **Rita** en siffra (0–9) i rutan till vänster.
    - 🔍 **Modellen analyserar** ritningen.
    - 🎯 **Resultat visas** i mittenrutan.
    - 📊 **Sannolikheter** visas till höger.
    """)


# Kolumn 1
col1, col2, col3 = st.columns([2, 2, 4])

with col1:
    st.markdown("<h4 style='width: 90%; text-align: center;'>🎨 Rita din siffra:</h4>", unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        display_toolbar=False,
    )
    
    
    #Knappar
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

#Canvas
if canvas_result.image_data is not None:
    try:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        processed = preprocess_canvas(img)
        st.session_state.processed = processed

        if processed is not None and processed.shape == (1, 784) and processed.sum() > 1000:
        
            prediction = st.session_state.model.predict(processed)[0]
            st.session_state.prediction = int(prediction)
        else:
            st.session_state.prediction = None
    except Exception as e:
        st.error(f"Fel vid prediktion: {e}")
        st.session_state.prediction = None
        st.session_state.processed = None
else:
    st.session_state.prediction = None
    st.session_state.processed = None


# Kolumn 2
with col2:
    st.markdown("<h4 style='width: 90%; text-align: center;'>🤖 Modellen gissar:</h4>", unsafe_allow_html=True)

    prediction = st.session_state.prediction
    if prediction is not None:
        st.markdown(
            f"""
            <div style="
                width: 280px;
                height: 280px;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #262730;
                color: white;
                font-size: 100px;
                border-radius: 0px;
                font-weight: bold;
                text-align: center;
                box-shadow: 0 0 10px rgba(0,0,0,0.4);
            ">
                {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                width: 280px;
                height: 280px;
                background-color: #262730;
                border-radius: 0px;
                box-shadow: 0 0 10px rgba(0,0,0,0.4);
            "></div>
            """,
            unsafe_allow_html=True
        )
        
        
# Kolumn 3
with col3:
    st.markdown("<h4 style='text-align: center;'>🤔 Hur säker är modellen?</h4>", unsafe_allow_html=True)

    if st.session_state.processed is not None:
        probs = model.predict_proba(st.session_state.processed)[0]
    else:
        probs = [0.0] * 10

    df_probs = pd.DataFrame({"Siffra": list(range(10)), "Sannolikhet": probs})

    chart = alt.Chart(df_probs).mark_bar().encode(
        x=alt.X("Siffra:O", title="Siffra", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Sannolikhet:Q", title="Sannolikhet", axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1])),
        tooltip=["Siffra", alt.Tooltip("Sannolikhet", format=".2%")]
    ).properties(width=300, height=400)

    st.altair_chart(chart, use_container_width=True)


# Info-ruta
st.info("💡 Tips: För bästa resultat – rita långsamt och tydligt.")