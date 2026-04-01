import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Loads the pre-split Kaggle dataset from the data folder."""
    print("Loading data...")
    # Paths are updated to look directly into the 'data' folder from the root
    train_df = pd.read_csv('data/train.txt', sep=';', names=['text', 'emotion'])
    test_df = pd.read_csv('data/test.txt', sep=';', names=['text', 'emotion'])
    return train_df, test_df

def main():
    # 1. Load Data
    train_df, test_df = load_data()
    
    X_train_raw = train_df['text']
    y_train = train_df['emotion']
    
    X_test_raw = test_df['text']
    y_test = test_df['emotion']

    # 2. Vectorization (Convert text to numbers)
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 3. Define the Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(max_iter=2000, dual=False),
        "Multinomial Naive Bayes": MultinomialNB()
    }

    # Make sure the models directory exists in the root folder
    os.makedirs('models', exist_ok=True)

    # Save the vectorizer first (we need this later to process new tweets)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Saved Vectorizer to models/tfidf_vectorizer.pkl")

    # 4. Train, Evaluate, and Save Each Model
    print("\n--- Starting Training Process ---")
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict on the unseen test data
        predictions = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, predictions)
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predictions))
        
        # Save the trained model to the models folder
        filename = model_name.replace(" ", "_").lower() + ".pkl"
        save_path = f'models/{filename}'
        joblib.dump(model, save_path)
        print(f"Saved {model_name} to {save_path}")

    print("\nAll models trained and saved successfully! 🚀")

if __name__ == "__main__":
    main()