# **Quora Question Pairs - Duplicate Detection**

This repository contains a comprehensive Jupyter Notebook aimed at analyzing the Quora Question Pairs dataset and developing machine learning models to identify semantically equivalent (duplicate) questions. The project encompasses thorough data exploration, preprocessing, advanced feature engineering, embedding generation, model training, and evaluation.

## **Authors**

* [Parit Kansal](https://www.github.com/ParitKansal)
* [Pankhuri Kansal](https://github.com/Pankhuri9026)

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
* LinearSVC
* Random Forest
* Extra Trees
* Gradient Boosting
* LightGBM
* XGBoost
* Naive Bayes
* AdaBoost

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

<details>
  <summary>📊 <strong>Model Performance Details</strong> — Click to expand</summary>

  |    | Classifier         | Metric Optimized   | Best Params                                                                                                |   Test Accuracy |   Test F1 Score |   Test Precision |   Test Recall |   Test Log Loss |
|---:|:-------------------|:-------------------|:-----------------------------------------------------------------------------------------------------------|----------------:|----------------:|-----------------:|--------------:|----------------:|
|  0 | XGBoost            | accuracy           | {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8}                             |        0.837065 |        0.824616 |         0.824792 |      0.824442 |        0.341579 |
|  1 | XGBoost            | f1                 | {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8}                             |        0.837065 |        0.824616 |         0.824792 |      0.824442 |        0.341579 |
|  2 | XGBoost            | precision          | {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8}                             |        0.837065 |        0.824616 |         0.824792 |      0.824442 |        0.341579 |
|  3 | XGBoost            | recall             | {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8}                             |        0.837065 |        0.824616 |         0.824792 |      0.824442 |        0.341579 |
|  4 | XGBoost            | log_loss           | {'n_estimators': 100, 'max_depth': 15, 'learning_rate': 0.1, 'subsample': 0.8}                             |        0.837065 |        0.824616 |         0.824792 |      0.824442 |        0.341579 |
|  5 | LogisticRegression | accuracy           | {'C': 10.0, 'solver': 'liblinear'}                                                                         |        0.752655 |        0.729537 |         0.734141 |      0.726223 |        0.472916 |
|  6 | LogisticRegression | f1                 | {'C': 10.0, 'solver': 'liblinear'}                                                                         |        0.752655 |        0.729537 |         0.734141 |      0.726223 |        0.472916 |
|  7 | LogisticRegression | precision          | {'C': 10.0, 'solver': 'liblinear'}                                                                         |        0.752655 |        0.729537 |         0.734141 |      0.726223 |        0.472916 |
|  8 | LogisticRegression | recall             | {'C': 10.0, 'solver': 'liblinear'}                                                                         |        0.752655 |        0.729537 |         0.734141 |      0.726223 |        0.472916 |
|  9 | LogisticRegression | log_loss           | {'C': 1.0, 'solver': 'liblinear'}                                                                          |        0.752655 |        0.729537 |         0.734141 |      0.726223 |        0.472916 |
| 10 | LinearSVC          | accuracy           | {'C': 1.0, 'max_iter': 1000}                                                                               |        0.754568 |        0.731899 |         0.736209 |      0.728733 |      inf        |
| 11 | LinearSVC          | f1                 | {'C': 1.0, 'max_iter': 1000}                                                                               |        0.754568 |        0.731899 |         0.736209 |      0.728733 |      inf        |
| 12 | LinearSVC          | precision          | {'C': 1.0, 'max_iter': 1000}                                                                               |        0.754568 |        0.731899 |         0.736209 |      0.728733 |      inf        |
| 13 | LinearSVC          | recall             | {'C': 1.0, 'max_iter': 1000}                                                                               |        0.754568 |        0.731899 |         0.736209 |      0.728733 |      inf        |
| 14 | RandomForest       | accuracy           | {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4, 'bootstrap': False} |        0.824434 |        0.81062  |         0.811473 |      0.809808 |        0.402194 |
| 15 | RandomForest       | f1                 | {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4, 'bootstrap': False} |        0.824434 |        0.81062  |         0.811473 |      0.809808 |        0.402194 |
| 16 | RandomForest       | precision          | {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4, 'bootstrap': False} |        0.824434 |        0.81062  |         0.811473 |      0.809808 |        0.402194 |
| 17 | RandomForest       | recall             | {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4, 'bootstrap': False} |        0.824434 |        0.81062  |         0.811473 |      0.809808 |        0.402194 |
| 18 | RandomForest       | log_loss           | {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4, 'bootstrap': False} |        0.824434 |        0.81062  |         0.811473 |      0.809808 |        0.402194 |
| 19 | GradientBoosting   | accuracy           | {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}                                                |        0.804878 |        0.791026 |         0.789798 |      0.792376 |        0.398173 |
| 20 | GradientBoosting   | f1                 | {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}                                                |        0.804878 |        0.791026 |         0.789798 |      0.792376 |        0.398173 |
| 21 | GradientBoosting   | precision          | {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}                                                |        0.804878 |        0.791026 |         0.789798 |      0.792376 |        0.398173 |
| 22 | GradientBoosting   | recall             | {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}                                                |        0.804878 |        0.791026 |         0.789798 |      0.792376 |        0.398173 |
| 23 | GradientBoosting   | log_loss           | {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}                                                |        0.804878 |        0.791026 |         0.789798 |      0.792376 |        0.398173 |
| 24 | AdaBoost           | accuracy           | {'n_estimators': 100, 'learning_rate': 0.5}                                                                |        0.705973 |        0.673538 |         0.681733 |      0.669478 |        0.554216 |
| 25 | AdaBoost           | f1                 | {'n_estimators': 100, 'learning_rate': 0.1}                                                                |        0.705973 |        0.673538 |         0.681733 |      0.669478 |        0.554216 |
| 26 | AdaBoost           | precision          | {'n_estimators': 100, 'learning_rate': 0.1}                                                                |        0.705973 |        0.673538 |         0.681733 |      0.669478 |        0.554216 |
| 27 | AdaBoost           | recall             | {'n_estimators': 100, 'learning_rate': 0.1}                                                                |        0.705973 |        0.673538 |         0.681733 |      0.669478 |        0.554216 |
| 28 | AdaBoost           | log_loss           | {'n_estimators': 50, 'learning_rate': 0.5}                                                                 |        0.705973 |        0.673538 |         0.681733 |      0.669478 |        0.554216 |
| 29 | ExtraTrees         | accuracy           | {'n_estimators': 50, 'max_depth': 15}                                                                      |        0.772739 |        0.747626 |         0.758789 |      0.741237 |        0.473426 |
| 30 | ExtraTrees         | f1                 | {'n_estimators': 50, 'max_depth': 15}                                                                      |        0.772739 |        0.747626 |         0.758789 |      0.741237 |        0.473426 |
| 31 | ExtraTrees         | precision          | {'n_estimators': 50, 'max_depth': 15}                                                                      |        0.772739 |        0.747626 |         0.758789 |      0.741237 |        0.473426 |
| 32 | ExtraTrees         | recall             | {'n_estimators': 50, 'max_depth': 15}                                                                      |        0.772739 |        0.747626 |         0.758789 |      0.741237 |        0.473426 |
| 33 | ExtraTrees         | log_loss           | {'n_estimators': 30, 'max_depth': 15}                                                                      |        0.772739 |        0.747626 |         0.758789 |      0.741237 |        0.473426 |
| 34 | LightGBM           | accuracy           | {'n_estimators': 150, 'num_leaves': 50, 'learning_rate': 0.1}                                              |        0.81736  |        0.803769 |         0.803378 |      0.804171 |        0.376051 |
| 35 | LightGBM           | f1                 | {'n_estimators': 150, 'num_leaves': 50, 'learning_rate': 0.1}                                              |        0.81736  |        0.803769 |         0.803378 |      0.804171 |        0.376051 |
| 36 | LightGBM           | precision          | {'n_estimators': 150, 'num_leaves': 50, 'learning_rate': 0.1}                                              |        0.81736  |        0.803769 |         0.803378 |      0.804171 |        0.376051 |
| 37 | LightGBM           | recall             | {'n_estimators': 150, 'num_leaves': 50, 'learning_rate': 0.1}                                              |        0.81736  |        0.803769 |         0.803378 |      0.804171 |        0.376051 |
| 38 | LightGBM           | log_loss           | {'n_estimators': 150, 'num_leaves': 50, 'learning_rate': 0.1}                                              |        0.81736  |        0.803769 |         0.803378 |      0.804171 |        0.376051 |
| 39 | NaiveBayes         | accuracy           | {}                                                                                                         |        0.692896 |        0.68637  |         0.689941 |      0.704235 |        3.20005  |
| 40 | NaiveBayes         | f1                 | {}                                                                                                         |        0.692896 |        0.68637  |         0.689941 |      0.704235 |        3.20005  |
| 41 | NaiveBayes         | precision          | {}                                                                                                         |        0.692896 |        0.68637  |         0.689941 |      0.704235 |        3.20005  |
| 42 | NaiveBayes         | recall             | {}                                                                                                         |        0.692896 |        0.68637  |         0.689941 |      0.704235 |        3.20005  |
| 43 | NaiveBayes         | log_loss           | {}                                                                                                         |        0.692896 |        0.68637  |         0.689941 |      0.704235 |        3.20005  |
</details>


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

## 📚 References

* Kaggle Quora Question Pairs Competition
* `fuzzywuzzy` Documentation
* `gensim` Word2Vec Tutorials
* Scikit-learn Evaluation Metrics Documentation
