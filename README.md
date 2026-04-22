![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# 📰 Fake News Detection — End-to-End NLP Project

> **IronHack · Week 4 · Day 3**
> A 3-day team project to build a binary text classifier that decides whether a news headline is **Fake (0)** or **Real (1)**.

---

## 👥 Team

| # | Member | Role | Main Deliverable |
|---|--------|------|------------------|
| 1 | **Sabeur** | Data & Preprocessing Lead | `clean_text(text) → processed_text` |
| 2 | **Philippe** | Feature Engineering & Modeling Lead | `model_pipeline.fit(X_train, y_train)` |
| 3 | **Joao** | Evaluation, Testing & Deployment Lead | `predictions = model.predict(X_test)` |

---

## 🗂️ Project Structure

```
fake-news-detection-nlp/
├── dataset/
│   ├── training_data_lowercase.csv          # label, headline (0=Fake, 1=Real)
│   └── testing_data_lowercase_nolabels.csv  # headline only — predictions to generate
├── doc/
│   └── Presentation template.pptx
├── main.ipynb                               # full end-to-end pipeline notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 End-to-End Pipeline

```
Raw Data ─► Data Cleaning ─► Preprocessing ─► Feature Engineering ─► Model Training ─► Evaluation ─► Predictions
 (csv)        (Sabeur)        (Sabeur)            (Philippe)            (Philippe)       (Joao)        (Joao)
```

| Step | Owner | Description |
|------|-------|-------------|
| 1. Data Loading & Cleaning | Sabeur | Load CSVs, fix formatting, drop duplicates / NaNs |
| 2. EDA | Sabeur | Class balance, headline length, word frequencies |
| 3. NLP Preprocessing | Sabeur | Tokenize · stopwords · punctuation · lemmatization → `clean_text()` |
| 4. Feature Engineering | Philippe | TF-IDF baseline (BoW / n-grams optional) |
| 5. Model Training | Philippe | Logistic Regression · Naïve Bayes · Linear SVM + GridSearchCV |
| 6. Evaluation | Joao | Accuracy · Precision · Recall · F1 · confusion matrix · error analysis |
| 7. Prediction | Joao | Apply pipeline to test set |
| 8. Output | Joao | `submission.csv` + persisted `model_pipeline.joblib` |

---

## 🔗 Integration Rules

- ✅ Sabeur defines `clean_text()` → **Philippe MUST use it** (no custom preprocessing).
- ✅ Philippe defines `vectorizer + model` → **Joao MUST reuse it** (no retraining).
- ❌ Do **NOT** call `.fit()` / `.fit_transform()` on test data — only `.transform()` and `.predict()`.

---

## 📅 3-Day Timeline

**Day 1 — Build the Foundation**
- Sabeur: load data, EDA, preprocessing pipeline.
- Philippe: TF-IDF setup, baseline model sanity check.

**Day 2 — Modeling & Evaluation**
- Philippe: train LR / NB / SVM, hyperparameter tuning.
- Joao: train/validation split, evaluation framework.

**Day 3 — Testing & Final Output**
- Joao: predictions on test set, error analysis.
- Team: review, finalize submission, prepare presentation.

---

## 🚀 Getting Started

### 1. Clone and create a virtual environment
```bash
git clone <repo-url>
cd fake-news-detection-nlp
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK resources (first run only)
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### 4. Run the notebook
```bash
jupyter notebook main.ipynb
```

---

## 🛠️ Tools & Libraries

`pandas` · `numpy` · `scikit-learn` · `nltk` · `matplotlib` · `seaborn` · `joblib` · `jupyter`

---

## 📊 Dataset

| File | Columns | Notes |
|------|---------|-------|
| `training_data_lowercase.csv` | `label`, `headline` | `0 = Fake`, `1 = Real` — used to train & validate |
| `testing_data_lowercase_nolabels.csv` | `headline` | No labels — predictions will be generated |

---

## 📦 Deliverables

1. **Python code** — well-documented notebook ([main.ipynb](main.ipynb)).
2. **Predictions** — `submission.csv` with predicted labels (`0` / `1`) in the original test order, same separator/format.
3. **Accuracy estimation** — validation metrics + cross-validation results.
4. **Presentation** — 10-minute team presentation (template in [doc/](doc/)).

---

## 💡 Optional Improvements

- 🔤 Word embeddings (Word2Vec / GloVe).
- 🤖 Transformer models (BERT / RoBERTa).
- 🌐 Streamlit web app for live demo.

---

🏆 **Teamwork makes the model work!**
