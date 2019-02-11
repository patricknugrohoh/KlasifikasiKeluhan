import joblib
import pandas as pd
import re
import string

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(path):
    clf = joblib.load(path)
    return clf

def load_vectorizer(path):
    vectorizer = joblib.load(path)
    return vectorizer

def url_remove(tweet):
    pattern = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    return re.sub(pattern, '', tweet)

def mention_remove(tweet):
    pattern = re.compile("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
    return re.sub(pattern, '', tweet)

def punctuation_remove(tweet):
    chars = set(string.punctuation)
    return ''.join(w for w in tweet if w not in chars)

def stopwords_remove(tweet):
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return remover.remove(tweet)

def stem(tweet):
    stemmer = StemmerFactory().create_stemmer()
    return stemmer.stem(tweet)

def preprocess_twitter(tweet):
    cleantext = url_remove(tweet)
    cleantext = mention_remove(cleantext)
    cleantext = punctuation_remove(cleantext)
    cleantext = stopwords_remove(cleantext)
    cleantext = stem(cleantext)

    return cleantext

def vectorize(tweet, option):
    vectorizer = load_vectorizer('../model/vectorizer/vec{}.joblib'.format(option))
    X = vectorizer.transform(tweet)
    return X

def classify(vector_tweet, vec_option, model_option):
    if model_option == 1:
        classifier = load_model('../model/tree/tree{}.joblib'.format(vec_option))
    elif model_option == 2:
        classifier = load_model('../model/svm/svm{}.joblib'.format(vec_option))
    elif model_option == 3:
        classifier = load_model('../model/mlp/mlp{}.joblib'.format(vec_option))
    return classifier.predict(vector_tweet)

def cek_keluhan(tweet, vec_option, model_option):
    vector_tweet = vectorize([tweet], vec_option)

    return classify(vector_tweet, vec_option, model_option)
