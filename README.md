```markdown
# 🎬 IMDB Movie Review Sentiment Analysis using RNN

A deep learning-based sentiment analysis system that classifies IMDB movie reviews as **positive** or **negative**, built using a Recurrent Neural Network (RNN) and deployed via a **Streamlit** web application.

---

## 🚀 Demo

🔗 **Live Demo (Optional)**: *Add HuggingFace/Streamlit Cloud link if deployed*

📷 **Screenshot**  
<img src="app_screenshot.png" alt="App UI" width="700"/>

---

## 📌 Key Features

- ✅ IMDB review sentiment classification (Positive / Negative)
- 🧠 Deep Learning Model: RNN trained on tokenized review text
- 💬 NLP Preprocessing: Tokenization, Padding, Sequencing
- 💾 Model saved in `.h5` format (Keras)
- 🌐 Streamlit App for real-time sentiment prediction
- 🧪 Reproducible setup via `rnnenv/` (virtual environment)

---

## 🧠 Model Architecture

- **Embedding Layer**
- **Recurrent Neural Network (Simple RNN / LSTM)**
- **Dense Output Layer with Sigmoid Activation**
- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy

---

## 📂 Project Structure

```

IMDB-Movies-Review-Analysis-/
├── rnnenv/                   # Virtual environment (Python)
├── app.py                   # Streamlit app for live prediction
├── model.h5                 # Trained RNN model (Keras)
├── Traning.ipynb            # Notebook for model training
└── .gitignore               # Files to ignore in Git

````

---

## 📥 Dataset

- **Source**: [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Classes**: Positive (1) or Negative (0)
- **Preprocessing**:
  - Lowercasing
  - Tokenization using Keras
  - Padding sequences to uniform length

---

## 🔧 Installation & Running Locally

1. **Clone the repo**

```bash
git clone https://github.com/PrinceGupta8/IMDB-Movies-Review-Analysis-.git
cd IMDB-Movies-Review-Analysis-
````

2. **Create a virtual environment** (or use provided `rnnenv/` if using the same OS)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. **Install requirements**

```bash
pip install -r requirements.txt  # You can export this using pip freeze > requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## ⚙️ Usage

* Input a movie review into the text box
* Click **Predict**
* View the **Sentiment Output**: `Positive` or `Negative`

---

## 📊 Results (If Available)

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | 85%+      |
| Loss      | \~0.35    |
| Inference | Real-time |

---

## 🚀 Future Improvements

* [ ] Replace RNN with BiLSTM or Transformer (BERT)
* [ ] Add confidence score
* [ ] Dockerize for production
* [ ] Integrate with REST API (Flask / FastAPI)
* [ ] Host on HuggingFace Spaces or Streamlit Cloud

---

## 🤝 Contributions

Contributions, ideas, and feedback are welcome! Feel free to open an issue or submit a pull request.

---

## 🧑‍💻 Author

**Prince Gupta**
📧 [princegupta995643@gmail.com](mailto:princegupta995643@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/prince-gupta-a8129a209/)
🔗 [GitHub](https://github.com/PrinceGupta8)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🌟 Show Your Support

If you find this project useful, please consider giving it a ⭐ on GitHub!

```
