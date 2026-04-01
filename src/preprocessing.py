import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only needs to be run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# 1. Emotion to Sentiment Mapping Function
def map_to_sentiment(emotion):
    """Maps Ekman emotions to standard sentiment categories."""
    emotion = str(emotion).strip().lower()
    
    positive_emotions = ['joy', 'happy', 'happiness'] # Adding variations just in case
    negative_emotions = ['anger', 'fear', 'sadness', 'disgust', 'sad', 'angry']
    
    if emotion in positive_emotions:
        return 'positive'
    elif emotion in negative_emotions:
        return 'negative'
    elif emotion == 'surprise':
        # Surprise can be tricky; often categorized as neutral or dropped for strict polarity
        return 'neutral' 
    else:
        return 'unknown'

# 2. Twitter Text Cleaning Function
def clean_tweet(text):
    """Cleans raw Twitter text for NLP processing."""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from hashtags (keeping the word)
    text = re.sub(r'\@\w+|\#','', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words (like 'the', 'is', 'in')
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin tokens into a single string
    return " ".join(filtered_tokens)


# --- Pipeline Execution ---

def main():
    print("Loading dataset...")
    # NOTE: Replace 'tec_dataset.csv' with the actual path to your downloaded file.
    # We are assuming the dataset has at least two columns: 'text' and 'emotion'
    try:
        df = pd.read_csv('tec_dataset.csv')
    except FileNotFoundError:
        print("Error: Could not find 'tec_dataset.csv'. Please update the file path.")
        # Creating a tiny mock dataframe just to show how it works if the file isn't there yet
        df = pd.DataFrame({
            'text': ["I am so #happy today! Best day ever :) https://link.com", 
                     "This traffic is making me furious @citytransit #angry",
                     "Wow, I didn't expect that ending! #surprise"],
            'emotion': ['joy', 'anger', 'surprise']
        })
        print("Using mock data for demonstration.\n")

    print("Mapping emotions to sentiment...")
    df['sentiment'] = df['emotion'].apply(map_to_sentiment)
    
    print("Cleaning tweet text...")
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    
    print("\n--- Processing Complete ---")
    print("Sample of the processed data:")
    print(df[['text', 'emotion', 'sentiment', 'cleaned_text']].head())

if __name__ == "__main__":
    main()