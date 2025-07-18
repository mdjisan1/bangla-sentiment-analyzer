
# 🇧🇩 Bangla Sentiment Analyzer (BERT + ONNX + LIME)

A web-based Bengali sentiment classifier with explainable AI support. This app uses a fine-tuned [Bangla BERT model](https://huggingface.co/sagorsarker/bangla-bert-base) exported to ONNX for fast inference. It classifies Bengali texts into harmful content categories and highlights the most influential words using **LIME**.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Setup Instructions](#setup-instructions)
7. [Usage](#usage)
8. [Model Details](#model-details)
9. [Explainable AI](#explainable-ai)
10. [Screenshots](#screenshots)
11. [Contributing](#contributing)
12. [License](#license)

---

## 🧩 Overview

This project aims to detect cyberbullying or offensive content in the Bengali language using state-of-the-art transformer-based NLP models. The fine-tuned model classifies text into one of five categories:

- `not bully`
- `religious`
- `sexual`
- `troll`
- `threat`

The project integrates:

- A BERT-based model fine-tuned on labeled Bengali text data
- ONNX runtime for fast inference
- LIME for explainable AI
- A full-stack Flask web application for interactive use

---

## ✨ Features

- ✅ Bengali language support with custom preprocessing
- ✅ Fast ONNX inference using fine-tuned BERT
- ✅ LIME-based word importance explanations
- ✅ Responsive, modern Flask web interface
- ✅ Educational and research-friendly architecture

---

## 🏗️ Architecture

```
User Input (Bengali text)
        │
        ▼
 [Preprocessing: clean, normalize]
        │
        ▼
 [Tokenizer: sagorsarker/bangla-bert-base]
        │
        ▼
 [ONNX InferenceSession → Prediction]
        │                └──► LIME Explainer (optional)
        ▼
  Predicted Label + Explanation
        │
        ▼
  Web Frontend Display (Flask)
```

---

## 📁 Project Structure

```
bangla-sentiment-analyzer/
├── app.py                    # Flask backend
├── BanglaBERT_ONNX.onnx      # ONNX exported model
├── templates/
│   └── index.html            # HTML UI
├── static/                   # Optional: CSS, JS
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation (this file)
```

---

## 📊 Dataset

The dataset used for training this model is publicly available here:  
🔗 [Bangla Text Dataset - Cyberbullying & Offensive Language](https://github.com/cypher-07/Bangla-Text-Dataset)

It includes labeled examples of Bengali text categorized under five types of cyberbullying and offensive language.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/mdjisan1/bangla-sentiment-analyzer.git
cd bangla-sentiment-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Visit: `http://localhost:5000`

---

## 🧠 Model Details

- **Base Model**: [`sagorsarker/bangla-bert-base`](https://huggingface.co/sagorsarker/bangla-bert-base)
- **Fine-tuned on**: Cyberbullying Dataset (5 labels)
- **Export Format**: ONNX
- **Tokenizer**: HuggingFace AutoTokenizer
- **Inference**: `onnxruntime` for low latency

---

## 🧠 Explainable AI

LIME (Local Interpretable Model-Agnostic Explanations) is used to interpret predictions.

- It perturbs the input text and observes changes in prediction
- Top influential words are extracted and shown with percentages

### Example:

> Input: `ওই হালার পুত এখন কি মদ খাওয়ার সময়`  
> Prediction: `troll`  
> Explanation:
- `হালার (38%)`
- `মদ (24%)`
- `পুত (15%)`

---

## 📷 Screenshots

> ⚠️ Replace placeholder images with actual screenshots.

### 1. 📊 Dataset Label Distribution
![Label Distribution](screenshots/Dataset%20Label%20Distribution.png)

### 2. 🔁 Confusion Matrix (Jupyter)
![Confusion Matrix](screenshots/Confusion%20Matrix.png)

### 3. 🔮 Prediction & Explanation (Jupyter)
![Jupyter](screenshots/Notebook%20Prediction.png)

### 4. 🌐 Web Interface – Prediction + LIME Explanation
![Web Result](screenshots/Web%20Interface%20with%20prediction.png)

---

## 🤝 Contributing

Want to improve or extend this project? You can:

- Add support for multi-label classification
- Integrate REST API or deploy to HuggingFace Spaces
- Improve UI with charts and animations
- Create REST endpoints or a Streamlit alternative

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙋‍♂️ Author

**Md Jisan Ahmed**  
📧 [jisan3325@gmail.com]  

---

> “Empowering Bengali NLP through Explainable AI” 🇧🇩
