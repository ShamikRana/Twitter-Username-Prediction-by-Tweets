import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import pickle
from flask import Flask, render_template, request

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load the trained model
# model = None
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
model = load_model('model.h5')

# Load the label encoder
label_encoder = None
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the tokenizer
tokenizer = None
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.replace('\n', ' ')
    text = text.lower()
    text = re.sub(r'\bhttp\w*\b', '', str(text))
    text = text.replace('amp',' ')
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text_tokens = text.split()
    filtered_tokens = [token for token in text_tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def stem_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    stemmed_text = ' '.join(stemmed_tokens)
    return stemmed_text

def predict_party(text):
    processed_text = preprocess_text(text)
    text_without_stopwords = remove_stopwords(processed_text)
    stemmed_text = stem_text(text_without_stopwords)
    
    encoded_text = tokenizer.texts_to_sequences([stemmed_text])
    padded_text = pad_sequences(encoded_text, maxlen=204)
    
    prediction = model.predict(padded_text)
    predicted_party = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_party[0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    predicted_party = predict_party(tweet)
    return render_template('predicted.html', tweet=tweet, predicted_party=predicted_party)


if __name__ == '__main__':
    app.run()

