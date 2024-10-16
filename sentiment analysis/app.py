from flask import Flask, request, render_template, jsonify
import joblib
import re
import pickle
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load pre-trained Word2Vec model and RandomForest model
with open("word2vec_model.pkl", "rb") as f:
    word2vec_model = pickle.load(f)
model = joblib.load('random_forest_model.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Tokenize and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens  # Returning tokens directly for Word2Vec

def get_average_word2vec(tokens, model, vector_size):
    """Convert tokens to word2vec by averaging the word vectors."""
    # Initialize an empty vector
    vec = np.zeros(vector_size)
    count = 0
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count != 0:
        vec /= count  # Get the average
    return vec

# Initialize Flask app
app = Flask(__name__)

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Define sentiment analysis route that handles form input
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Get input text from the form
        input_text = request.form['text']
        
        # Preprocess the input text
        tokens = preprocess_text(input_text)
        
        # Get average word2vec representation
        vector_size = word2vec_model.vector_size  # Get the size of word vectors
        word2vec_vector = get_average_word2vec(tokens, word2vec_model, vector_size)
        
        # Reshape the vector to match the model's expected input format
        word2vec_vector = word2vec_vector.reshape(1, -1)
        
        # Predict sentiment using the loaded RandomForest model
        prediction = model.predict(word2vec_vector)
        
        # Return the prediction result to be displayed on the webpage
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('index.html', result=sentiment)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
