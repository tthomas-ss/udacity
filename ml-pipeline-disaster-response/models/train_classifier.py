import sys
import pandas as pd
import re
import nltk
import time
from sklearn.externals import joblib
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Load data from SQLite database stored at database_filepath (str).

    Args:
    database_filepath (str) -- path to SQLite database, including file name.

    Returns:
    X (pandas.DataFrame) -- Messages to be classified.
    Y (panas.DataFrame) -- Target categories.
    category_names (list) -- list of target category names
    """
    # Create SQLAlchemy engine and read table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    # Split the data into text and labels
    X = df['message']
    Y = df.iloc[:, 4:]
    # Create a list of category names/labels
    category_names = list(Y)

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize input text, including lammatization and stop-word removal

    Args:
    text (str) -- a string to tokenize

    Returns:
    tokens (list) -- a list of words tokenized from text.
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words("english")
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build a ML pipeline with a RandomForrestClassifier predictor.  The workspace is running on a single-CPU instance,
    and training takes forever.

    Returns:
    model (sklearn.model_selection._search.GridSearchCV) -- Pipeline with multi output random forest classifier.
    """

    # ML pipeline for transforming and classifying.
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tf_idf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                         ])
    # Grid search parameter dictionary
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'tf_idf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [100, 300]
    }

    grid_model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate trained model on testdata

    Args:
    model (sklearn.pipeline.Pipeline) -- the trained model
    X_test (pandas.DataFrame) -- test dataset features
    Y_test (pandas.DataFrame) -- test dataset target
    category_names (str) -- names of the target labels

    """
    # Make predictions
    Y_preds = model.predict(X_test)
    # Print classification report
    print(classification_report(Y_preds, Y_test.values, target_names=category_names))
    print("**** Accuracy scores *****\n")
    # Print accuracy scores
    for i, cat in enumerate(list(Y_test)):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:, i], Y_preds[:, i]))


def save_model(model, model_filepath):
    """
    Save best estimator (sklearn.pipeline.Pipeline) from to model_filepath (str).
    """
    joblib.dump(model, model_filepath, compress=3)


def load_model(model_filepath):
    """Return saved model (sklearn.pipeline.Pipeline) from model_filepath (str)."""
    model = joblib.load(model_filepath)
    return model


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, load_saved_model = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # Training takes hours - I have added the option to load a saved model.
        if load_saved_model == 'Y':
            model = load_model(model_filepath)

        # Build and train grid-search.
        else:
            print('Building model...')
            model = build_model()
            print('Training model...')
            model.fit(X_train, Y_train)
            # Make it a Pipeline object from the best estimator found in grid search.
            # So that it can be saved/loaded as the actual best pipeline.
            model = model.best_estimator_

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()