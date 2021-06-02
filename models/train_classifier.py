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
    """
    The function to read the data from the database.
    
    Inputs:
            database_filepath:  File path to the database to save the dataframe
            
    Outputs: 
        X             :     Dataframe containg the X data to train the ML pipeline
        Y             :     Dataframe containg the X data to train the ML pipeline
        category_names:     labels of the categries columns
    """
    import os
    print(os.getcwd())
    eng = 'sqlite:///' + database_filepath
    engine = create_engine(eng)
    df = pd.read_sql_table('message', engine)
    X = df['message']
    Y = df.drop(columns= ['id','message', 'original', 'genre'], axis = 1)
    # as small test version just uncomment the following:
    #X = X.head(100)
    #Y = Y.head(100)

    category_names = Y.columns
    return X, Y , category_names

#tokenizing function
def tokenize(text):
    """
    The function to tokenize and lemmatize the text.
    
    Inputs:
        text:                the text which needs to be tokenized
        
    Outputs:
        tokens:              tokens which can be used in machine learning
     
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# pipeline bildup function
def build_model():
    """
    The function to build the machine learning pipeline using NLTK with GridSearchCV. 
    
    Output:
        pipeline:                 contains the pipeline model with the GridSearchCV Parameters
        
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    # the parameter check is perform with the grid search method und the best results are used in the pipeline
    parameters = {'clf__estimator__leaf_size': (20, 30, 50),
                  'clf__estimator__n_neighbors': (5,10,36),
                  'clf__n_jobs': (1,2),
                  'vect__max_df': (0.5, 0.7,1),
                  'vect__ngram_range': ((1, 1), (1, 2),(2,2))}
    

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

# Model evaluation function
def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function to evaluate the trained model using the test data. 
    Inputs: 
        model :                   the machine learning model
        X_test:                   the test split of the X data
        Y_test:                   the test split of the Y data
        category_names:           labels of the categries columns
    Output:
        classification_report:    the report indicating the f1 score for each category
        
    """
    y_pred = model.predict(X_test)
    labels = np.unique(Y_test)
    y_pred =pd.DataFrame(np.array(y_pred),columns=category_names)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print("\nBest Parameters:", model.best_params_)
    model.parameters = model.best_params_
# function to xport the model as a pickle file
def save_model(model, model_filepath):
    """
    The function to save the optimal model as a pickle file. 
    Inputs: 
        model :                   the machine learning model
        model_filepath:           File path to the .pkl data       
    """
    import pickle

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

# the main function to perform the ML pipeline sarting with asking the filepathes
def main():
    """
    The function to run the pipeline including building the model, train, evaluate and save it.      
    """
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
