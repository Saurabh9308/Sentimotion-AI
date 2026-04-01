import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    f1_score, roc_curve, auc, ConfusionMatrixDisplay
)

# 1. SETUP FOLDERS (Automatic Creation)
base_results_path = os.path.join("results", "sentiment_analysis_results")
model_folders = ["linear_svm", "logistic_regression", "multinomial_naive_bayes"]

for folder in model_folders:
    os.makedirs(os.path.join(base_results_path, folder), exist_ok=True)

# 2. LOAD DATA & MODELS
# Yahan wahi cleaned test data use karna jo tumne Colab mein nikala tha
# Example ke liye main CSV se load kar raha hoon (ensure path is correct)
try:
    # Maan lo tumne test_data save kiya tha, warna ye logic use karo:
    # (Testing ke liye hum original data ka ek chota part use kar rahe hain)
    path_to_data = "training.1600000.processed.noemoticon.csv"
    columns = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(path_to_data, encoding='latin-1', names=columns).sample(10000, random_state=42)
    df['target'] = df['target'].replace(4, 1)
    
    # Preprocessing
    df['text'] = df['text'].str.lower().replace(r'[^a-z\s]', '', regex=True)
    X_test_raw = df['text']
    y_test = df['target']

    # Load Vectorizer
    vectorizer = joblib.load(os.path.join("models", "sentiment_analysis_models", "tfidf_vectorizer.pkl"))
    X_test = vectorizer.transform(X_test_raw)

    # Models Map
    models_to_test = {
        "linear_svm": "linear_svm_model.pkl",
        "logistic_regression": "logistic_regression_model.pkl",
        "multinomial_naive_bayes": "multinomial_nb_model.pkl"
    }

    plt.figure(figsize=(10, 8)) # For Combined ROC Curve

    for folder_name, model_file in models_to_test.items():
        print(f"Evaluating {folder_name}...")
        
        # Load Model
        model_path = os.path.join("models", "sentiment_analysis_models", model_file)
        model = joblib.load(model_path)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # --- A. Save Metrics (Text File) ---
        report = classification_report(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics_file = os.path.join(base_results_path, folder_name, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Model: {folder_name}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        # --- B. Save Confusion Matrix (Image) ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        plt.title(f'Confusion Matrix - {folder_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(base_results_path, folder_name, "confusion_matrix.png"))
        plt.close()

        # --- C. ROC Curve Data ---
        # Note: LinearSVC doesn't have predict_proba, using decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plotting on the combined chart
        plt.figure(1) # Go back to the combined ROC plot
        plt.plot(fpr, tpr, lw=2, label=f'{folder_name} (AUC = {roc_auc:.2f})')

    # --- D. Finalize & Save Combined ROC Curve ---
    plt.figure(1)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve - Sentiment Analysis')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(base_results_path, "combined_roc_comparison.png"))
    plt.close()

    print(f"\n✅ Sab kuch save ho gaya hai: {base_results_path} folder check karo!")

except Exception as e:
    print(f"❌ Error: {e}")