# **Quora Question Pairs - Duplicate Detection**

This repository contains a comprehensive Jupyter Notebook aimed at analyzing the Quora Question Pairs dataset and developing machine learning models to identify semantically equivalent (duplicate) questions. The project encompasses thorough data exploration, preprocessing, advanced feature engineering, embedding generation, model training, and evaluation.

## **Authors**

* [Parit Kansal](https://www.github.com/ParitKansal)
* [Pankhuri Kansal](https://github.com/Pankhuri9026)

## 📌 Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Setup & Installation](#setup--installation)
4. [Notebook Workflow](#notebook-workflow)
5. [Feature Engineering Details](#feature-engineering-details)
6. [Modeling and Evaluation](#modeling-and-evaluation)
7. [Usage](#usage)
8. [Requirements](#requirements)
9. [Contributing](#contributing)
10. [License](#license)
11. [References](#references)

## 📖 Overview

With Quora receiving over 100 million monthly visitors, users frequently post similar questions phrased differently. This project leverages Natural Language Processing (NLP) techniques to determine whether pairs of questions are duplicates, thereby minimizing redundancy and enhancing information retrieval.

Key learning outcomes of this project include:

* Preprocessing and cleaning raw question text
* Engineering features using token-level patterns, string similarity metrics, and word embeddings
* Training various machine learning models
* Evaluating models using standard classification metrics

## 📂 Dataset

**Source:** [Kaggle - Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs)

**Dataset Fields:**

* `id`: Unique identifier for each question pair
* `qid1`, `qid2`: Unique identifiers for individual questions
* `question1`, `question2`: Text content of each question
* `is_duplicate`: Target label (1 if duplicate, 0 otherwise)

## ⚙️ Setup & Installation

### Dataset Download

```bash
# Configure Kaggle API
kaggle competitions download -c quora-question-pairs -p data/
unzip data/train.csv.zip -d data/
```

### (Optional) Download Pre-trained Word2Vec Embeddings

```bash
gdown https://somesource/GoogleNews-vectors-negative300.bin.gz -O data/
gzip -d data/GoogleNews-vectors-negative300.bin.gz
```

## 📒 Notebook Workflow

### 1. Library Imports

Includes essential libraries such as:

* `numpy`, `pandas`, `matplotlib`, `seaborn`
* `re`, `string`, `nltk`, `spacy`
* `fuzzywuzzy`, `gensim`, `scikit-learn`, `xgboost`, `pickle`

### 2. Exploratory Data Analysis (EDA)

* Handling missing values
* Analyzing distribution of duplicate vs. non-duplicate questions
* Examining frequently occurring questions

### 3. Data Preprocessing

* Converting text to lowercase
* Removing emojis, HTML tags, and URLs
* Replacing contractions and special characters
* Eliminating punctuation and extra spaces

## 🧠 Feature Engineering Details

### Length-Based Features

* Individual question lengths
* Absolute difference in token counts
* Longest common substring and respective ratios

### Token-Based Features

* Count and ratio of shared words
* Overlap ratios based on stopwords
* Matching of first and last words

### Fuzzy Matching Features

Utilized the `fuzzywuzzy` package to compute:

* `fuzz.ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`

### Word2Vec Embedding Features

* Applied pre-trained GoogleNews Word2Vec model on both questions
* Averaged word vectors were used as feature representations

## 📊 Modeling and Evaluation

### Classifiers Implemented

* Logistic Regression
* Random Forest
* Extra Trees
* Gradient Boosting
* K-Nearest Neighbors (KNN)
* XGBoost
* Naive Bayes

### Evaluation Metrics

Models were evaluated using the following metrics:

* Accuracy
* F1-score
* Precision
* Recall
* Logarithmic Loss (Log Loss)

### Hyperparameter Tuning

Hyperparameters for each classifier were optimized using a validation dataset created via stratified splitting of the training data. This ensured the robustness and generalizability of the selected parameters, reducing the risk of overfitting.

### Final Evaluation Results

| Classifier | Best Parameters                                                                  | Test Accuracy | F1 Score   | Precision  | Recall     | Log Loss   |
| ---------- | -------------------------------------------------------------------------------- | ------------- | ---------- | ---------- | ---------- | ---------- |
| XGBoost    | `{'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8}` | **0.8371**    | **0.8246** | **0.8248** | **0.8244** | **0.3416** |

> ✅ **Conclusion:** Among all tested models, **XGBoost consistently delivered superior performance** across all key evaluation metrics and is recommended as the optimal classifier for this task.

> 📘 **Note:** For comprehensive performance details of each model, including validation results and confusion matrices, please consult the accompanying Jupyter Notebook.

## 📦 Requirements

* Python 3.8 or later
* `numpy==1.23.5`
* `pandas`
* `scikit-learn`
* `gensim`
* `fuzzywuzzy`
* `xgboost`
* `kaggle`
* `gdown`

## 🤝 Contributing

We welcome contributions in the following areas:

* Enhancing preprocessing techniques
* Introducing new feature engineering methods
* Integrating deep learning-based models

Please submit a pull request with a clear description of your changes.

## 📝 License

This project is licensed under the [MIT License](LICENSE). Portions of the code have been adapted from sources licensed under [GNU GPL v3.0](https://github.com/ParitKansal/quora-question-pairs/blob/main/LICENSE).

## 📚 References

* Kaggle Quora Question Pairs Competition
* `fuzzywuzzy` Documentation
* `gensim` Word2Vec Tutorials
* Scikit-learn Evaluation Metrics Documentation
