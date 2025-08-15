from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
import pickle
from keras.models import load_model
import os

app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load chatbot files
try:
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json', encoding='utf-8').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Create bag of words array"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the intent class of the sentence"""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Get a random response from the matching intent"""
    if not intents_list:
        return "I didn't understand that. Could you rephrase?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    if request.method == 'POST':
        user_message = request.form['message']
        
        if model is None:
            return jsonify({'response': "Chatbot model not loaded properly. Please check server logs."})
        
        ints = predict_class(user_message)
        response = get_response(ints, intents)
        return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)