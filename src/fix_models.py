import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION ---
# Check your sidebar in VS Code: Is it 'sentimental' or 'sentiment'? 
# Based on your screenshot, it is 'sentimental_analysis_models'
EMOTION_DIR = "models/emotion_analysis_models"
SENTIMENT_DIR = "models/sentimental_analysis_models"

def fix_folder(folder_path, vectorizer_name, data_file):
    print(f"\n--- Checking Folder: {folder_path} ---")
    
    # Create the folder if it somehow doesn't exist
    if not os.path.exists(folder_path):
        print(f"Creating missing folder: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)

    # 1. Load the raw text data
    data_path = f"data/{data_file}"
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}. Please check your 'data' folder.")
        return

    print(f"Loading data from {data_path}...")
    # Using low_memory=False to avoid warnings with large text files
    df = pd.read_csv(data_path, sep=';', names=['text', 'emotion'])
    
    # 2. Re-fit a new vectorizer
    print(f"Re-fitting vectorizer for {vectorizer_name}...")
    new_vec = TfidfVectorizer(max_features=10000, stop_words='english')
    new_vec.fit(df['text'].astype(str)) # Ensure text is string
    
    # 3. Save the new vectorizer
    vec_save_path = os.path.join(folder_path, vectorizer_name)
    joblib.dump(new_vec, vec_save_path)
    print(f"✅ Successfully saved: {vec_save_path}")

# --- EXECUTION ---
# Fix Emotion Folder
fix_folder(EMOTION_DIR, "tfidf_vectorizer.pkl", "train.txt")

# Fix Sentiment Folder (Note the 'al' in sentimental)
fix_folder(SENTIMENT_DIR, "vectorizer.pkl", "train.txt")

print("\n🚀 Cleanup complete! Try running your app now:")
print("streamlit run app/streamlit_app.py")