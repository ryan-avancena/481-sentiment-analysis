from flask import Flask, request, render_template

import create_models
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

# add some form of error checking here

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
        
        """ GETTING ALL THE VARIABLES WE NEED FOR THE PREDICTION """

        # Prior Probabilities
        prior_probabilities = model.class_log_prior_

        # Conditional Probabilities (Feature log probabilities)
        feature_log_prob = model.feature_log_prob_

        # Get the TF-IDF values for each word in the cleaned sentence
        words = tf_idf.get_feature_names_out()
        tf_idf_values = review_tf_idf.toarray()[0]

        top_words = []
        for idx, word in enumerate(words):
            if tf_idf_values[idx] > 0:  # Include only relevant words
                log_prob_positive = feature_log_prob[1, idx]  # Log prob for positive class
                log_prob_negative = feature_log_prob[0, idx]  # Log prob for negative class
                top_words.append((word, tf_idf_values[idx], log_prob_positive, log_prob_negative))

        # Sort by TF-IDF value (descending) and take top 10
        top_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:10]
        rounded_top_words = [
            (word, round(tfidf_value, 3), round(log_prob_positive, 3), round(log_prob_negative, 3)) 
            for word, tfidf_value, log_prob_positive, log_prob_negative in top_words
        ]

        # Compute the log probabilities for each class
        log_probs = prior_probabilities + np.sum(feature_log_prob * review_tf_idf.toarray(), axis=1)

        # posterior_probs = np.exp(log_probs)
        # normalized_probs = posterior_probs - np.max(posterior_probs)   

        normalized_probs = np.exp(log_probs - np.max(log_probs))  # Stability adjustment
        normalization_constant = normalized_probs.sum()
        final_posterior = normalized_probs/normalization_constant
        # posterior_probs /= posterior_probs.sum()

        """ ROUNDING THE VALUES """
        features = features
        prediction = prediction
        predicted_probabilities = np.round(predicted_probabilities, 3).tolist()  # Rounding probabilities to 3 decimal places
        prior_probabilities = np.round(prior_probabilities, 3).tolist()          # Rounding prior probabilities
        log_probs = np.round(log_probs, 3).tolist()                              # Rounding log probabilities
        # posterior_probs = np.round(posterior_probs, 3).tolist()                  # Rounding posterior probabilities
        final_posterior = np.round(final_posterior*100, 3).tolist()
    else:
        prediction = "No input provided."
    return render_template('prediction.html', 
                            features=features,
                            prediction=prediction,
                            predicted_probabilities=predicted_probabilities,
                            cleaned_sentence=cleaned_sentence,
                            prior_probabilities=prior_probabilities,
                            top_words=rounded_top_words,
                            log_probs=log_probs,
                            normalized_probs=normalized_probs,
                            normalization_constant=normalization_constant,
                            final_posterior=final_posterior
                           )

if __name__ == '__main__':
    app.run(debug=True)


# @app.route("/")
# def hello_world():
#     return "<p>Hello,world!</p>"