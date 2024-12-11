# CPSC 481 - Sentiment Analysis with McDonalds
![image](https://github.com/user-attachments/assets/6ba588b4-2e89-4c2e-be50-c91b8c8cf56e)

## Project Overview
This project analyzes customer reviews to gain insights into customer experiences at McDonald's restaurants worldwide. By leveraging a Naive Bayes Classifier, we aim to identify trends in customer sentiment and provide actionable insights to enhance service quality across various locations.

## Repository Contents
- **`app.py`**: A Flask application that provides mathematical analysis and explains how the Naive Bayes Classifier predicts the sentiment of a review.
- **`create_models.py`**: A Python script for creating the models used in the Flask application, including the Naive Bayes Classifier and TF-IDF matrix.
- **`481_mcdonalds_sentiment_analysis.ipynb`**: Interactive Jupyter Notebook utilized for initial development.
- **`mcdonalds_reviews.csv`**: Dataset utilized for the the entire project.
- **`mcdonalds_sentiment_model.pkl`**: `pkl` file to import the model into `app.py`.
- **`tfidf_vectorizer.pkl`**: `pkl` file to import our tf-idf matrix into `app.py`.
- **`/static`**: holds `.css` files for Flask initialization of our webpage.
- **`/venv`**: Virtual environment for library imports.
- **`/templates`**: holds `.html` files for Flask intialization of our webpage.
- **`requirements.txt`**: All libraries utilized in this project.


## How to Run the Application
Follow these steps to set up and run the application:

## How to Run
1. Clone this repository.
2. Make sure `pip` is installed.
3. If pip is not installed, follow the [pip installation guide](https://pip.pypa.io/en/stable/installation/).
4. Activate the virtual environment
   - For MacOS users: Enter 'source ./venv/Scripts/activate' in the terminal.
   - For Windows/Linux users: Enter './venv/Scripts/activate' in the terminal.
5. Enter 'pip install -r requirements.txt' in the terminal to install all the dependencies used.
6. Run 'app.py' to host the webpage locally (http://127.0.0.1:5000).
   
(Optional) - If there are any errors with loading the models, run the 'create_models.py' file to build the NB Classifer and TF-IDF Matrix.
   
