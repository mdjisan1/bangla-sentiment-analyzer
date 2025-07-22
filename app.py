from flask import Flask, request, render_template
import re
import os
import gdown
import unicodedata
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer

app = Flask(__name__)

MODEL_PATH = 'BanglaBERT_ONNX.onnx'
GDRIVE_FILE_ID = '19SsIltlVzM_CIt1gy5CbFei5fsfRylut'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading ONNX model using gdown...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists.")

download_model()

# Model and tokenizer
TOKENIZER_NAME = 'sagorsarker/bangla-bert-base'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

MODEL_PATH = 'BanglaBERT_ONNX.onnx'
session = ort.InferenceSession(MODEL_PATH)

label_map = {
    0: 'not bully',
    1: 'religious',
    2: 'sexual',
    3: 'troll',
    4: 'threat'
}

def preprocess_bangla_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFC', text)
    return text

def predict(text):
    processed = preprocess_bangla_text(text)
    enc = tokenizer(processed, truncation=True, padding='max_length', max_length=128, return_tensors='np')
    ort_inputs = {
        session.get_inputs()[0].name: enc['input_ids'].astype(np.int64),
        session.get_inputs()[1].name: enc['attention_mask'].astype(np.int64)
    }
    logits = session.run(None, ort_inputs)[0]
    pred_id = int(np.argmax(logits, axis=1)[0])
    return label_map.get(pred_id, 'unknown'), logits

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def lime_predict(texts):
    processed = [preprocess_bangla_text(t) for t in texts]
    enc = tokenizer(processed, truncation=True, padding='max_length', max_length=64, return_tensors='np')
    ort_inputs = {
        session.get_inputs()[0].name: enc['input_ids'].astype(np.int64),
        session.get_inputs()[1].name: enc['attention_mask'].astype(np.int64)
    }
    logits = session.run(None, ort_inputs)[0]
    return softmax(logits)

def get_lime_explanation(text, top_k=3):
    explainer = LimeTextExplainer(class_names=list(label_map.values()), split_expression=r'\s+')
    explanation = explainer.explain_instance(text, lime_predict, num_features=top_k, num_samples=300)
    top_words = explanation.as_list()
    total = sum(abs(score) for _, score in top_words) or 1e-6
    token_contributions = [f"{word} ({int(100 * abs(score) / total)}%)" for word, score in top_words if word.strip()]
    return token_contributions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['text']
        sentiment, _ = predict(input_text)
        explanations = get_lime_explanation(input_text)
        return render_template('index.html', input_text=input_text, sentiment=sentiment, explanations=explanations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
