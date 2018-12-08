'''
This file contains utility functions for the DNB Data Science Nanodegree capstone project.
'''

import nltk
from nltk.corpus import stopwords
from itertools import cycle

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_curve


nltk.download('stopwords')


def plot_confusion_matrix(y_test, predictions):
    """Plots a confusion matrix."""

    cm = confusion_matrix(y_test, predictions)
    strings = np.asarray([['True negatives', 'False positives'], ['False negatives', 'True positives']])

    labels = (np.asarray(["{} : {}".format(string, value)
                          for string, value in zip(strings.flatten(),
                                                   cm.flatten())])
              ).reshape(2, 2)

    ax = sb.heatmap(cm, annot=labels, fmt='', cmap="YlGnBu", cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    plt.show()


def plot_roc(y_test, predicted_proba, name):
    """Plots a ROC curve"""
    # calculate the fpr and tpr for all thresholds of the classification

    if predicted_proba.shape[1] == 2:
        preds = predicted_proba[:, 1]
    else:
        preds = predicted_proba

    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic - {}'.format(name))
    plt.plot(fpr, tpr, 'b', linewidth=2, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--', linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def threshold_search(y_true, y_proba, min_threshold=0.1, max_threshold=1.0, step=0.01):
    """
    Function to search a probability prediction for the best cutoff-value (highest F1 score)
    """
    # To handle both sklearn predict_proba and Keras predictions
    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    best_threshold = 0
    best_score = 0

    for threshold in np.arange(min_threshold, max_threshold, step):
        threshold = np.round(threshold, 2)
        score = f1_score(y_true=y_true, y_pred=(y_proba > threshold).astype(int))
        print("F1 score at threshold {0} is {1}".format(threshold, score))
        if score > best_score:
            best_threshold = threshold
            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}
    print('Best F1 {} at {}'.format(best_score, best_threshold))

    return search_result


def clean_text(text, lower_stop=False):
    """
    Function to clean text, standardizing to lower case and removing stop words.

    :param text: An input string to be cleaned
    :return: cleaned text
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    longnums_regex = '\d{6,}'
    logchar_regex = r'((\w)\2{2,})'

    detected_urls = re.findall(url_regex, text)
    detected_longnums = re.findall(longnums_regex, text)
    detected_longchar = re.findall(logchar_regex, text)

    for url in detected_urls:
        text = text.replace(url, ".")

    for num in detected_longnums:
        text = text.replace(num, ".")

    for char in detected_longchar:
        text = text.replace(char[0], ".")

    if lower_stop:
        stops = set(stopwords.words("english"))
        text = text.lower()
        text = [w.lower() for w in text if not w in stops]
        text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=?]", " ", text)

    return text


def load_data(path, clean=False, lower_stop=False):
    """
    Convenience function to load and clean text

    :param path: path to source data
    :param clean_text: set to true if
    :return:
    """
    print('Loading questions...')
    train = pd.read_csv('{}/train.csv'.format(path))
    print('Done loading train - Loading test')
    test = pd.read_csv('{}/test.csv'.format(path))
    print('Done loading test')

    if clean:
        print('Cleaning train')
        train['question_text'] = train['question_text'].map(lambda x: clean_text(x, lower_stop=lower_stop))
        print('Cleaning test')
        test['question_text'] = test['question_text'].map(lambda x: clean_text(x, lower_stop=lower_stop))

    corpus = pd.concat([train['question_text'], test['question_text']])

    return train, test, corpus


def load_embeddings(path='../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
    """
    Loads word embeddings from file
    :param path: Path to word embeddings
    :return: Dict - key = word, value = word vector representation
    """
    print('Loading word vectors...')
    word2vec = {}
    with open(path) as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vec = np.asarray(values[1:], dtype='float32')
            except Exception as e:
                print('Word: {} - {}'.format(word, e))

            word2vec[word] = vec

    print('Found %s word vectors.' % len(word2vec))
    return word2vec


def load_embedding_matrix(word_ix_map, word_vector_map, vocabulary_size, embedding_dim):
    """
    Creating a lookup matrix for word embeddings ,using the vocabulary in word_ix_map
    :param word_ix_map: map of words resulting from tokenization
    :param word_vector_map: word embeddings
    :return: filled matrix
    """
    print('Filling pre-trained embeddings...')
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_ix_map.items():
        if i < vocabulary_size:
            embedding_vector = word_vector_map.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def plot_precision_recall_curve(y_test, predicted_proba, best_thresh):
    """
    Plots the precision recall curve with information
    :param y_test: Actual y values
    :param predicted_proba: predicted probabilities
    :param best_thresh: integer - the best cutoff-point
    :return: n/a
    """
    precision, recall, thresholds = precision_recall_curve(y_test, predicted_proba)
    _f1 = f1_score(y_test, (predicted_proba > best_thresh).astype(int))

    plt.figure(figsize=(14, 8))
    plt.title("Precision and Recall curve")
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0, 1.01]);
    plt.xlim([0, 1.01]);
    plt.xlabel('Recall', fontsize=12);
    plt.ylabel('Precision', fontsize=12);

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - best_thresh))

    plt.annotate('Current cutoff set here (F1={:,.2f}) \nPrecision:{:,.2f}\nRecall:{:,.2f}'. \
                 format(_f1, precision[close_default_clf], recall[close_default_clf]),
                 size=12, xy=(recall[close_default_clf], precision[close_default_clf]),
                 xytext=(recall[close_default_clf], precision[close_default_clf] + 0.1),
                 arrowprops=dict(arrowstyle="->"))
    plt.show()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Plotting precision and recall values per threshold.
    Thanks to Kevin Arvai for supplying this here:
    https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(10, 7))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.show()

