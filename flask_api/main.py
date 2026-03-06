import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
from dotenv import load_dotenv
import os
load_dotenv()


app = Flask(__name__)
CORS(app)

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not','but','however','no','yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f'Error in preprocessing comment: {e}')
        return comment
    
# def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
#     aws_path = os.getenv("AWS_INSTANCE_IP")
#     mlflow.set_tracking_uri(f"http://{aws_path}:5000")
#     client = MlflowClient()
#     model_uri = f'models:/{model_name}/{model_version}'
#     model = mlflow.pyfunc.load_model(model_uri)
#     with open(vectorizer_path, 'rb') as file:
#         vectorizer = pickle.load(file)
        
#     return model, vectorizer

def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except Exception as e:
        raise
    
model, vectorizer = load_model('./lgbm_model.pkl','./tfidf_vectorizer.pkl')


@app.route('/')
def home():
    return 'Welcome to our flask api'

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    # print('i am the comment: ', comments)
    # print('i am the comment type: ', type(comments))

    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}),500
    
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)
        
        
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, debug = True)