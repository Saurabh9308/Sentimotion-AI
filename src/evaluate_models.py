import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Configuration
MODELS_DIR = 'models'
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# The exact filenames we saved in the previous step
model_names = [
    'logistic_regression', 
    'linear_svm', 
    'multinomial_naive_bayes'
]

def load_test_data():
    """Loads the test data and vectorizes it using our saved TF-IDF model."""
    print("Loading test data and vectorizer...")
    # Load raw text
    test_df = pd.read_csv(f'{DATA_DIR}/test.txt', sep=';', names=['text', 'emotion'])
    X_test_raw = test_df['text']
    y_test = test_df['emotion']
    
    # Load vectorizer and transform text
    vectorizer = joblib.load(f'{MODELS_DIR}/tfidf_vectorizer.pkl')
    X_test = vectorizer.transform(X_test_raw)
    
    return X_test, y_test

def save_individual_results(name, model, X_test, y_test):
    """Generates and saves the Confusion Matrix and Classification Report."""
    print(f"Generating details for {name.replace('_', ' ').title()}...")
    folder = os.path.join(RESULTS_DIR, name)
    os.makedirs(folder, exist_ok=True)
    
    y_pred = model.predict(X_test)
    classes = model.classes_
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {name.replace("_", " ").title()}')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(f'{folder}/confusion_matrix.png', dpi=300)
    plt.close()

    # 2. Classification Report
    report = classification_report(y_test, y_pred, target_names=classes)
    with open(f'{folder}/metrics.txt', 'w') as f:
        f.write(report)

def plot_combined_roc(loaded_models, X_test, y_test):
    """Generates a combined Micro-Averaged ROC curve for multi-class data."""
    print("Generating combined ROC Curve comparison...")
    plt.figure(figsize=(10, 8))
    
    # We need the class names to binarize the labels for the ROC curve
    first_model = list(loaded_models.values())[0]
    classes = first_model.classes_
    y_test_bin = label_binarize(y_test, classes=classes)
    
    for name, model in loaded_models.items():
        formatted_name = name.replace("_", " ").title()
        
        # Get decision scores depending on the model type
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)
            
        # Compute Micro-average ROC curve and ROC area
        # This treats all classes equally and calculates global true/false positive rates
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{formatted_name} (Micro-AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Global)')
    plt.ylabel('True Positive Rate (Global)')
    plt.title('Multi-Class Model Comparison: Micro-Averaged ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f'{RESULTS_DIR}/combined_roc_comparison.png', dpi=300)
    plt.close()

def main():
    # Create main results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    X_test, y_test = load_test_data()
    loaded_dict = {}
    
    print("\n--- Starting Evaluation ---")
    for name in model_names:
        try:
            model = joblib.load(f'{MODELS_DIR}/{name}.pkl')
            loaded_dict[name] = model
            save_individual_results(name, model, X_test, y_test)
        except FileNotFoundError:
            print(f"Warning: Could not find model file for {name}. Skipping...")

    if loaded_dict:
        plot_combined_roc(loaded_dict, X_test, y_test)
        print(f"\nSuccess! All metrics and beautiful high-res images are saved in the '{RESULTS_DIR}' folder. 🚀")
    else:
        print("\nError: No models were loaded. Did you run the training script first?")

if __name__ == "__main__":
    main()