from flask import Flask, request, render_template
import pickle
import numpy as np
import joblib
import re

app = Flask(__name__)

'''
Pretrained Models: model, tf_idf

model: a pre-trained multinomial naive-bayes classifier
tf_idf: a pre-trained 5000 word tf-idf matrix to vectorize our text
'''

model = joblib.load('./mcdonalds_sentiment_model.pkl')
tf_idf = joblib.load('./tfidf_vectorizer.pkl')

'''
Function: clean_text(text)

Input: review as a string

Output: cleaned review (lowercase, removed punctuation, stemming)

We're taking the request and cleaning the review to ensure that we're just working with the words.
We could remove stopwords aswell but they aren't included in our TF-IDF Matrix already.
'''

def clean_text(text):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]','',cleaned_text)
    return cleaned_text

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def make_prediction():
    features = list(request.form.values())
    # print("Features", features)
    if len(features) > 0:
        cleaned_sentence = clean_text(features[0])
        review_tf_idf = tf_idf.transform([cleaned_sentence])
        predicted_probabilities = model.predict_proba(review_tf_idf)
        prediction = model.predict(review_tf_idf)[0]
        # print(cleaned_sentence)
        # print(predicted_probabilities)
    else:
        prediction = "No input provided."
    return render_template('prediction.html', 
                           prediction=prediction,
                           predicted_probabilities=predicted_probabilities,
                           cleaned_sentence=cleaned_sentence
                           )

if __name__ == '__main__':
    app.run(debug=True)

# @app.route("/")
# def hello_world():
#     return "<p>Hello,world!</p>"