![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Fake News Detection - End-to-End NLP Project

A small team project for IronHack week 4. We had three days to build a classifier that reads a news headline and decides whether it's fake (0) or real (1). The whole pipeline lives in one notebook so it's easy to follow from start to finish.

## The team

A three-person team, split by stage of the pipeline:

| Role | What they own |
|------|--------------|
| Data lead | Data, cleaning, EDA, the `clean_text()` function |
| Modelling lead | Vectoriser, models, hyperparameter tuning |
| Evaluation lead | Train/val split, evaluation, predictions, final submission |

## Results

After tuning, a **Linear SVM on TF-IDF features** came out on top.

| Metric | Score |
|--------|-------|
| Validation accuracy | **92.6 %** |
| Precision | 0.89 |
| Recall | 0.97 |
| F1 | 0.93 |
| Baseline accuracy (untuned) | 94.24 % |
| Best CV accuracy (after tuning) | 94.94 % |

Takeaway: even a simple TF-IDF + linear model is surprisingly strong at this task *if* the preprocessing is done carefully.

## Repo layout

```
fake-news-detection-nlp/
├── dataset/
│   ├── training_data_lowercase.csv          # label + headline (0=fake, 1=real)
│   └── testing_data_lowercase_nolabels.csv  # headline only - what we predict
├── doc/
│   ├── Presentation template.pptx
│   ├── fake_news_detection_presentation.pptx
│   └── challenge_insight_slide.pptx
├── main.ipynb                               # everything happens here
├── requirements.txt
├── .gitignore
└── README.md
```

## How the pipeline flows

```
raw csv  ->  clean  ->  preprocess  ->  vectorise  ->  train  ->  evaluate  ->  predict
```

| Step | Owner | What happens |
|------|-------|--------------|
| Load & clean | Data lead | Read the CSVs, drop duplicates, fix types |
| EDA | Data lead | Class balance, headline length, common words |
| Preprocess | Data lead | Lowercase, strip punctuation, tokenise, lemmatise -> `clean_text()` |
| Features | Modelling lead | TF-IDF (with BoW / n-grams as alternatives) |
| Train | Modelling lead | Logistic Regression, Naive Bayes, Linear SVM + GridSearchCV |
| Evaluate | Evaluation lead | Accuracy, precision, recall, F1, confusion matrix, error analysis |
| Predict | Evaluation lead | Run the pipeline on the test set |
| Submit | Evaluation lead | Save `submission.csv` and the trained `model_pipeline.joblib` |

## Rules we agreed on

- `clean_text()` is the only preprocessing function. Everyone imports and reuses it.
- Vectoriser + classifier go into one sklearn `Pipeline`. Evaluation reuses that exact object.
- On test data: `.transform()` and `.predict()` only. Never `.fit()` - that would leak.

## Three days, roughly

**Day 1 - foundations**
- Load the data, do some EDA, write the preprocessing.
- End the day with TF-IDF + a baseline model to make sure the cleaned data behaves.

**Day 2 - modelling**
- Train LR / NB / SVM, then tune the best one with cross-validation.
- Set up the train/val split and the evaluation helpers.

**Day 3 - testing and wrap-up**
- Run predictions on the test set and dig into the errors.
- Review, finalise the submission and prepare the slides.

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

## Biggest challenge & key insight

**Biggest challenge.** Fake news is written to look real - same language, same style, same clickbait patterns. Add sarcasm, evolving topics and noisy labels and it gets messy fast. That's why we put so much effort into preprocessing.

**Key insight.** Even a simple NLP pipeline detects fake news well when the text is cleaned properly, the features are sensible, and the model is evaluated honestly. Fancy models are not always the answer.

## What we hand in

1. The notebook ([main.ipynb](main.ipynb)).
2. `submission.csv` with one prediction per test headline, in the original order, same separator and format.
3. A short note on how well we expect the model to do (validation metrics + cross-validation results).
4. A 10-minute presentation (slides in [doc/](doc/)).

## If we'd had more time

- Try word embeddings (Word2Vec or GloVe).
- A pretrained transformer (BERT, RoBERTa).
- Wrap it in a small Streamlit app for live demos.

Teamwork makes the model work.
