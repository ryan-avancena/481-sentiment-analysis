from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

''' 
Overview: In this file, we'll be creating the models used in our Flask application.

Functions:
preprocess_text(text):
data_cleaning(dataset):

'''

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # words = [porter.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def data_cleaning(dataset):
    mcd_reviews = dataset
    cleaned_mcd_reviews = mcd_reviews[['review','rating']]
    cleaned_mcd_reviews = cleaned_mcd_reviews[
        (cleaned_mcd_reviews['rating'] == '1 star') |
        (cleaned_mcd_reviews['rating'] == '2 stars') |
        (cleaned_mcd_reviews['rating'] == '4 stars') |
        (cleaned_mcd_reviews['rating'] == '5 stars')
    ]

    cleaned_mcd_reviews.replace({'5 stars': 'positive', '1 star': 'negative',
                                '2 stars': 'negative', '4 stars': 'positive'}, inplace=True)

    cleaned_mcd_reviews.columns = ['text', 'sentiment']

    cleaned_mcd_reviews['cleaned_text'] = (
        cleaned_mcd_reviews['text']
        .str.lower()
        .str.replace(f"[{string.punctuation}]", "", regex=True)
        .apply(preprocess_text)
    )

    return cleaned_mcd_reviews

if __name__ == '__main__':
    mcd_reviews = pd.read_csv('mcdonalds_reviews.csv',encoding='ISO-8859-1')
    cleaned_mcd_reviews = data_cleaning(mcd_reviews)

    X = cleaned_mcd_reviews['cleaned_text']
    y = cleaned_mcd_reviews['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    y_pred = nb.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    joblib.dump(nb, 'mcdonalds_sentiment_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')