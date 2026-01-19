# AI-Generated Text Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Transformer](https://img.shields.io/badge/Model-Transformer-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Status-Academic%20Project-informational)

An AI-based system for detecting **AI-generated text** versus **human-written text** using transformer-based Natural Language Processing models. The project is developed and deployed using **Google Colab** and exposed publicly via **Ngrok**.

---

## 📌 Project Overview

With the rapid adoption of large language models, identifying AI-generated content has become an important research problem. This project implements a **binary text classification system** trained on both AI-generated and human-written samples.

The project is intended for **academic research, experimentation, and demonstration purposes**. Model training and web deployment are performed inside notebook environments rather than via a traditional local server setup.

---

## 🚀 Features

- Human-written vs AI-generated text classification  
- Transformer-based models (BERT / DeBERTa)  
- Google Colab notebooks for training and experimentation  
- Web-based GUI deployed using Ngrok  
- Pre-trained model files included  
- Clean and modular academic project structure  

---

## 🧠 Model Details

- **Architecture:** Transformer (BERT / DeBERTa)  
- **Task:** Binary sequence classification  
- **Framework:** PyTorch with Hugging Face Transformers  
- **Optimizer:** AdamW  
- **Loss Function:** Cross-Entropy Loss  

---

## ▶️ Usage (Google Colab + Ngrok)

This project **does not run as a local web application** using `python app.py`.

Instead, the web interface is launched from **Google Colab**, and **Ngrok** is used to create a public, clickable URL.

### Steps to Run

1. Open the GUI deployment notebook:
   ```
   notebooks/gui_deployment.ipynb
   ```

2. Run all cells in the notebook:
   - The Flask app is started inside the notebook  
   - Ngrok authenticates and creates a public tunnel  

3. After execution, Colab outputs a **clickable Ngrok URL**, for example:
   ```
   https://xxxx-xx-xx-xx.ngrok-free.app
   ```

4. Open the link in a browser to access the web application.

### Deployment Code Snippet

```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")

public_url = ngrok.connect(5000)
print(f"Your app is live at: {public_url}")

app.run(port=5000)
```

> Each run generates a **temporary URL** that expires when the Colab session ends.

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## ⚠️ Limitations

- Ngrok URLs are temporary and session-based  
- Requires Google Colab to redeploy the web interface  
- Performance may degrade on very short text inputs  
- Detection accuracy depends on training data diversity  
- Not guaranteed to generalize to unseen future AI models  

---

## 📌 Future Enhancements

- Permanent cloud deployment (Hugging Face Spaces / Render / AWS)  
- Multi-class classification (AI model attribution)  
- Model explainability and attention visualization  
- REST API for external integration  

---

## 🧑‍💻 Author

**Mridupawan Gogoi**

---

## 📜 License

This project is licensed under the **MIT License**.
