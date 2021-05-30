# the ML Pipeline
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords

# function for loading data from the sqlite database
def load_data(database_filepath):
    import os
    print(os.getcwd())
    eng = 'sqlite:///' + database_filepath
    engine = create_engine(eng)
    df = pd.read_sql_table('message', engine)
    X = df['message']
    Y = df.drop(columns= ['id','message', 'original', 'genre'], axis = 1)

    category_names = Y.columns
    return X, Y , category_names

#tokenizing function
def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# pipeline bildup function
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = {'clf__estimator__leaf_size': 30,
                  'clf__n_jobs': 1,
                  'tfidf__use_idf': False,
                  'vect__max_df': 0.5,
                  'vect__max_features': None,
                  'vect__ngram_range': (1, 1)}
    
    
    return pipeline 

# Model evaluation function
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    labels = np.unique(Y_test)
    y_pred =pd.DataFrame(np.array(y_pred),columns=category_names)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], y_pred[col],labels=labels))

# function to xport the model as a pickle file
def save_model(model, model_filepath):
    import pickle

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

# the main function to perform the ML pipeline sarting with asking the filepathes
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
