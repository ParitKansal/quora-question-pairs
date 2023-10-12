# **Quora Question Pairs**

The goal of this project is to predict which of the provided pairs of questions contain two questions with the same meaning.

Link - [GoogleColab](https://github.com/ParitKansal/quora-question-pairs/blob/main/notebook.ipynb)

## **Authors**

- [@ParitKansal](https://www.github.com/ParitKansal)
- [@Pankhuri9026](https://github.com/Pankhuri9026)
## **Description**
In this project, we aim to address a common challenge on Quora: the proliferation of similar questions. With over 100 million monthly visitors, it's common for users to ask questions that have already been posed in slightly different ways. This redundancy can lead to inefficiencies as seekers struggle to find the best answers and writers feel compelled to respond to multiple versions of the same question.

Our goal is to leverage advanced natural language processing techniques to classify whether pairs of questions are duplicates or not. By doing so, we aim to streamline the process of finding high-quality answers, ultimately enhancing the experience for Quora writers, seekers, and readers alike.
## **Dataset**
Download dataset for custom training - [Dataset](https://drive.google.com/file/d/1ge8lHgEk9BSrkRZJEfbLSbYCTGQOwG6b/view?usp=drive_link)
## Data fields
- id - the id of a training set question pair
- qid1, qid2 - unique ids of each question (only available in train.csv)
- question1, question2 - the full text of each question
- is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.
## **Steps**
### Import Libraries
- numpy
- pandas
- matplotlib
- seaborn
- re
- string
- spacy
- nltk
- distance
- fuzzywuzzy
- gensim
- sklearn
- pickle

### EDA

- **Null Value :**  
  Analysis of the questions that contain null values. If the number of rows with null values is significantly smaller than the total number of rows, it may be reasonable to delete those rows.

- **Duplicate vs Non-Duplicate Analysis :**  
  Calculate the number of question pairs identified as duplicates and those deemed non-duplicates. Then, analyze the frequency of repeated questions and the number of occurrences.

### Preprocessing

- **Lower Case :**  
  Convert all the questions in the dataset to lowercase. This ensures uniformity in the text data and facilitates comparison and analysis.
- **Remove Emoji :**  
  Remove all emojis present in the questions in the dataset. Emojis do not contribute to the semantic meaning of the text and can introduce noise in the analysis.


- **Remove HTML :**  
  Remove all HTML links in the questions. HTML tags are not relevant for text analysis and can be safely removed.


- **Remove URL :**  
  Remove all URLs from the questions. URLs often do not contribute to the semantic content of the text and can be considered noise.

- **Replace Certain Special Characters with Their String Equivalents :**  
   Replace specific special characters with their string equivalents. This ensures that special characters are treated consistently during text processing.
  - '%'         ->      ' percent'
  - '$'         ->      ' dollar '
  - '₹'         ->      ' rupee '
  - '€'         ->      ' euro '
  - '@'         ->      ' at '
  - '[math]'    ->      ''

- **Replace Contractions :**  
  Replace all contractions with their expansions. This helps standardize the text and avoids ambiguity in interpretation.
  - "ain't" -> "am not"
  - "aren't" -> "are not" 
     etc.

- **Remove Punctuations :**  
  Remove all punctuation marks present in the questions. Punctuation marks typically do not contribute to the semantic meaning of the text and can be safely discarded.

### Feature Engineering

- **len :**
    - **q1_len** : length of question1
    - **q2_len** : length of question2

- **fetch_length_features :**
    - **q1_token_no** : no of tokens in question1
    - **q2_token_no** : no of tokens in question2
    - **abs_len_diff** : Absolute length features (q1_token_no - q2_token_no)
    - **mean_len** : Average Token Length of both Questions
    - **longest_substr** : Longest common substring
    - **longest_substr_ratio_min** : Ratio of longest common substring to min question len
    - **longest_substr_ratio_max** : Ratio of longest common substring to max question len.

- **fetch_token_features :**
    - **common_words** : no of common words/tokens
    - **total_words** : total words/tokens (q1_tokens + q2_tokens)
    - **ratio_of_common_to_total** : ratio of common tokens to total tokens (common_words / total_words)
    - **cwc_min** : the ratio of the number of common words to the length of the smaller question
    - **cwc_max** : the ratio of the number of common words to the length of the larger question
    - **csc_min** : the ratio of the number of common stop words to the smaller stop word count among the two questions
    - **csc_max** : the ratio of the number of common stop words to the larger stop word count among the two questions
    - **ctc_min** : the ratio of the number of common tokens to the smaller token count among the two questions
    - **ctc_max** : the ratio of the number of common tokens to the larger token count among the two questions
    - **last_word_eq** : 1 if the last word in the two questions is same, 0 otherwise
    - **first_word_eq** : 1 if the first word in the two questions is same, 0 otherwise

- **fetch_fuzzy_features :**
    - **fuzz_ratio** : fuzz.QRatio from fuzzywuzzy
    - **fuzz_partial_ratio** : fuzz.partial_ratio from fuzzywuzzy
    - **token_sort_ratio** :  fuzz.token_sort_ratio from fuzzywuzzy
    - **token_set_ratio** : fuzz.token_set_ratio from fuzzywuzzy**
    
### word2vec
- Applied pretained model of word2vec of GoogleNews on question1, question2 columns of the preprocessed data.
- Download dataset after applying preprocessing and word2vec- [Processed Dataset](https://github.com/ParitKansal/quora-question-pairs/blob/main/Processed_Dataset.csv)

### Modeling
- **Type of model selection :**  
  Perform train_test_split from sklearn.model_selection.  
  Perform Random Forest Classifier from sklearn.ensemble and fit the dataset to train the model. On testing the model, the accuracy obtained is 0.772.  
  Perform SVM classifier from sklearn library and fit the dataset to train the model. On testing the model, the accuracy obtained is 0.705.  
  Since, the accuracy of the Random Forest model is better than the SVM model, therefore we move forward with the Random Forest Model.

- **Hyperparameter Tuning of Random Forest Model :**
  - **Tuning and training of model to determine best parameter :**
      Use GridSearchCV from sklearn.model_selection to cross validate 5 times and hyper parameterize the parameters to determine the best of the parameters.
      The best values of these parameters are obtained as:
      - n_estimators : 115
      - criterion : 'log_loss'
      - max_depth : None
      - max_features : 0.5
      - max_samples : 1.0

- **Final Modelling :**  
  Perform Random Forest Classifier from sklearn.ensemble and fit the dataset to train the model based on the before estimated best values of the parameter and test the model to determine its accuracy. The accuracy obtained is  .  

  Download trained model : [Final Model](https://drive.google.com/file/d/1ge8lHgEk9BSrkRZJEfbLSbYCTGQOwG6b/view?usp=drive_link)
## Deployment of Model in Jupyter Notebook

- Download dataset after applying preprocessing and word2vec- [Processed Dataset](https://drive.google.com/file/d/1-0FVCRRCBv2kMAibJqtGo4B7ua54g-N8/view?usp=sharing)  
- Run the following comands in python notebook
  ```bash
  df = pd.read_csv("link_of_downloaded_Processes_Dataset")
  ```
  ```bash
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']), df['is_duplicate'], test_size=0.25, random_state=0)
  ```
- Download trained model : [Final Model](https://drive.google.com/file/d/1ge8lHgEk9BSrkRZJEfbLSbYCTGQOwG6b/view?usp=drive_link)
- Run the following comands in python notebook after previous commands
  ```bash
  import pickle
  model = pickle.load(open('link_of_downoaded_model', 'rb'))
  ```
  ```bash
  y_pred = model.predict(X_test)
  ```
  ```bash
  from sklearn.metrics import accuracy_score, confusion_matrix
  print("Accuracy : ", accuracy_score(y_pred,y_test))
  print("Confusion Matrix : ")
  print(confusion_matrix(y_pred, y_test))
  ```

## License

[GNU General Public License v3.0](https://github.com/ParitKansal/quora-question-pairs/blob/main/LICENSE)
