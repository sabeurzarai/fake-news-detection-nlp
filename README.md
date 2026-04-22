![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Fake News Detection - End-to-End NLP Project

A small team project for IronHack week 4. We had three days to build a classifier that reads a news headline and decides whether it's fake (0) or real (1). The whole pipeline lives in one notebook so it's easy to follow from start to finish.

## The team

| Member | What they own |
|--------|--------------|
| Sabeur | Data, cleaning, EDA, the `clean_text()` function |
| Philippe | Vectoriser, models, hyperparameter tuning |
| Joao | Train/val split, evaluation, predictions, final submission |

## Repo layout

```
fake-news-detection-nlp/
├── dataset/
│   ├── training_data_lowercase.csv          # label + headline (0=fake, 1=real)
│   └── testing_data_lowercase_nolabels.csv  # headline only - what we predict
├── doc/
│   └── Presentation template.pptx
├── main.ipynb                               # everything happens here
├── requirements.txt
├── .gitignore
└── README.md
```

## How the pipeline flows

```
raw csv  ->  clean  ->  preprocess  ->  vectorise  ->  train  ->  evaluate  ->  predict
             Sabeur     Sabeur          Philippe       Philippe   Joao         Joao
```

| Step | Owner | What happens |
|------|-------|--------------|
| Load & clean | Sabeur | Read the CSVs, drop duplicates, fix types |
| EDA | Sabeur | Class balance, headline length, common words |
| Preprocess | Sabeur | Lowercase, strip punctuation, tokenise, lemmatise -> `clean_text()` |
| Features | Philippe | TF-IDF (with BoW / n-grams as alternatives) |
| Train | Philippe | Logistic Regression, Naive Bayes, Linear SVM + GridSearchCV |
| Evaluate | Joao | Accuracy, precision, recall, F1, confusion matrix, error analysis |
| Predict | Joao | Run the pipeline on the test set |
| Submit | Joao | Save `submission.csv` and the trained `model_pipeline.joblib` |

## Rules we agreed on

- Sabeur's `clean_text()` is the only preprocessing function. Philippe and Joao import and reuse it.
- Vectoriser + classifier go into one sklearn `Pipeline` that Philippe builds. Joao uses that exact object.
- On test data: `.transform()` and `.predict()` only. Never `.fit()` - that would leak.

## Three days, roughly

**Day 1 - foundations**
- Sabeur loads the data, does some EDA and writes the preprocessing.
- Philippe ends the day setting up TF-IDF + a baseline model to make sure the cleaned data behaves.

**Day 2 - modelling**
- Philippe trains LR / NB / SVM, then tunes the best one with cross-validation.
- Joao sets up the train/val split and the evaluation helpers.

**Day 3 - testing and wrap-up**
- Joao runs predictions on the test set and digs into the errors.
- The whole team reviews, finalises the submission and prepares the slides.

## Getting it running

Clone the repo, set up a virtual environment, install the deps:

```bash
git clone <repo-url>
cd fake-news-detection-nlp
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

NLTK needs a couple of resources the first time:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

Then open the notebook:

```bash
jupyter notebook main.ipynb
```

## What we use

`pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `joblib`, `jupyter`.

## The dataset

| File | Columns | Notes |
|------|---------|-------|
| `training_data_lowercase.csv` | `label`, `headline` | 0 = fake, 1 = real. We train and validate on this. |
| `testing_data_lowercase_nolabels.csv` | `headline` | No labels - we generate them. |

## What we hand in

1. The notebook ([main.ipynb](main.ipynb)).
2. `submission.csv` with one prediction per test headline, in the original order, same separator and format.
3. A short note on how well we expect the model to do (validation metrics + cross-validation results).
4. A 10-minute presentation (template in [doc/](doc/)).

## If we'd had more time

- Try word embeddings (Word2Vec or GloVe).
- A pretrained transformer (BERT, RoBERTa).
- Wrap it in a small Streamlit app for live demos.

Teamwork makes the model work.
