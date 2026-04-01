Since you’ve built a hybrid system (Classical ML + Deep Learning), your README needs to highlight that **architectural sophistication**. This is what will impress recruiters and other developers.

# 🧠 Sentimotion-AI: Hybrid NLP Analysis Pipeline

**Sentimotion-AI** is a sophisticated Natural Language Processing (NLP) application that provides a dual-layered analysis of text. By combining **Classical Machine Learning** for efficient sentiment detection and **Deep Learning Transformers** for emotional nuance, it offers a comprehensive 360-degree view of human expression.

[Live Demo Link](https://sentimotion-ai-gfx7vgrbm7cr93gisapazy.streamlit.app/) ---

## 🏗️ The Hybrid Architecture

The project follows a "best-of-both-worlds" approach to balance speed and accuracy:

### 1\. Sentiment Branch (Classical ML)

  * **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
  * **Models:** Comparison of Linear SVM, Logistic Regression, and Multinomial Naive Bayes.
  * **Efficiency:** High-speed inference with minimal computational overhead.

### 2\. Emotion Branch (Deep Learning)

  * **Engine:** Hugging Face `transformers` pipeline.
  * **Model:** `distilbert-base-uncased-emotion` (Fine-tuned DistilBERT).
  * **Accuracy:** Captures semantic context to identify 6 core emotions: **Joy, Love, Anger, Sadness, Fear, and Surprise**.

-----

## 🚀 Key Features

  * **Real-time Dual Analysis:** Processes sentiment and emotion simultaneously.
  * **Interactive Sidebar:** Dynamic model selection for the sentiment backend.
  * **Visual Confidence Mapping:** Real-time bar charts showing the probability distribution of emotions.
  * **Production-Ready:** Deployed via Streamlit Cloud with optimized caching for model loading.

-----

## 🛠️ Tech Stack

  * **Language:** Python 3.9+
  * **Web Framework:** Streamlit
  * **Deep Learning:** Transformers (Hugging Face), PyTorch
  * **Machine Learning:** Scikit-Learn, Joblib
  * **Data Handling:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn

-----

## 📂 Project Structure
```text
.
├── .gitignore                  # Prevents bulky files (.venv, __pycache__) from being tracked
├── requirements.txt            # Production dependencies
├── app/
│   └── streamlit_app.py        # Main Streamlit logic & UI
├── models/
│   ├── emotion_analysis_models/ # Legacy or intermediate emotion models
│   └── sentiment_analysis_models/ # Serialized .pkl models & vectorizers
├── src/
│   ├── fix_models.py           # Utility scripts
│   ├── train_models.py         # Model training scripts
│   └── preprocessing.py        # Data cleaning & tokenization
├── results/                    # Output logs, confusion matrices, and metrics
└── README_images/              # (Recommended) Place for images used in this README
```

-----

## ⚙️ Local Setup

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/Saurabh9308/Sentimotion-AI.git
    cd Sentimotion-AI
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App:**
    ```bash
    streamlit run app/streamlit_app.py
    ```

-----

## 👨‍💻 Author

**Saurabh**
*Full-Stack Developer | ML Enthusiast*
[LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/saurabh-kadtan-558811293) | [GitHub](https://www.google.com/search?q=https://github.com/Saurabh9308)
