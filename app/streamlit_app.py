import streamlit as st
import joblib
import os
import pandas as pd
import warnings
from transformers import pipeline

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- PATH CONFIG ----------------
# This ensures the app finds your models regardless of where it's hosted
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion + Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODELS (CACHED) ----------------

@st.cache_resource
def load_deep_learning_emotion():
    """
    Loads a State-of-the-Art Deep Learning model for Emotion.
    """
    # This downloads from Hugging Face on first run (~260MB)
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

@st.cache_resource
def load_sentiment_assets(model_filename):
    """
    Loads local .pkl files for Sentiment.
    """
    base_path = os.path.join(BASE_DIR, "models", "sentiment_analysis_models")
    
    model_path = os.path.join(base_path, f"{model_filename}.pkl")
    vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")
    
    # Check if files exist before loading
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Could not find model or vectorizer in {base_path}")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

sentiment_models = {
    "SVM": "linear_svm_model",
    "Logistic Regression": "logistic_regression_model",
    "Naive Bayes": "multinomial_nb_model"
}

selected_sentiment = st.sidebar.selectbox("Sentiment Model", list(sentiment_models.keys()))

# ---------------- LOADING LOGIC ----------------
try:
    with st.spinner("Waking up the models..."):
        emotion_pipeline = load_deep_learning_emotion()
        s_model, s_vec = load_sentiment_assets(sentiment_models[selected_sentiment])
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Check that your folder structure matches: models/sentiment_analysis_models/")
    st.stop()

# ---------------- UI ----------------
st.title("🧠 Emotion & Sentiment Analyzer")
st.markdown("Combined Architecture: **Deep Learning (Emotion)** + **Classical ML (Sentiment)**")

text = st.text_area("Enter text to analyze", height=150, placeholder="Type a tweet here...")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner('Thinking...'):
            # 1. Sentiment Prediction
            s_input = s_vec.transform([text])
            s_pred = s_model.predict(s_input)[0]

            # 2. Emotion Prediction
            emotion_results = emotion_pipeline(text)[0] 
            top_emotion = emotion_results[0]['label']
            
        # -------- DISPLAY RESULTS --------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Result")
            # Handling common labels: 1/4 (Twitter sentiment) or strings
            if str(s_pred).lower() in ["1", "4", "positive", "pos"]:
                st.success("POSITIVE ✅")
            else:
                st.error("NEGATIVE ❌")

        with col2:
            st.subheader("Emotion Result")
            emojis = {
                "joy": "😄", "love": "❤️", "anger": "😠",
                "sadness": "😢", "fear": "😨", "surprise": "😲"
            }
            emoji = emojis.get(top_emotion, "📝")
            st.info(f"{top_emotion.upper()} {emoji}")

        # -------- CONFIDENCE BREAKDOWN --------
        st.divider()
        st.subheader("📊 Deep Learning Emotion Confidence")

        df_probs = pd.DataFrame(emotion_results)
        df_probs.columns = ["Emotion", "Confidence"]
        df_probs = df_probs.sort_values("Confidence", ascending=False)

        st.bar_chart(df_probs.set_index("Emotion"))