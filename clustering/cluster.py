import re
import json
import pandas as pd
import streamlit as st
import nltk
import numpy as np
from doc_gathering.collector import JSON_FILE
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class Cluster:

    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def __init__(self):
        self.docs = []

    @st.cache_data
    def load_file(_self):
        if not JSON_FILE.exists():
            st.error("Data Source file is missing")
            raise ValueError("Data source file is missing")

        print("Loading json file ...")
        with open(JSON_FILE) as f:
            docs = json.load(f)

        categories = pd.Series([d["category"] for d in docs]).value_counts()
        st.sidebar.write("Dataset:", categories.to_dict())
        return docs

    @st.cache_data
    def preprocess(_self, text):
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        tokens = [
            Cluster.stemmer.stem(t)
            for t in tokens
            if t not in Cluster.STOP_WORDS and len(t) > 2
        ]
        return " ".join(tokens)

    @st.cache_data
    def train_model(_self, docs):
        """Using k = 3 for three different categories"""
        texts = [_self.preprocess(d["text"]) for d in docs]
        vectorizer = TfidfVectorizer(max_features=2000)
        X = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        df = pd.DataFrame(
            {
                "text": [d["text"][:100] + "..." for d in docs],
                "category": [d["category"] for d in docs],
                "cluster": clusters,
            }
        )
        print(df)
        return kmeans, vectorizer, X, df

    def predict_cluster(self, query: str, kmeans, vectorizer: TfidfVectorizer):
        query = self.preprocess(query)
        vec = vectorizer.transform([query])
        distances = kmeans.transform(vec)[0]
        cluster = kmeans.predict(vec)[0]
        print(kmeans.predict(vec)[0])
        confidence = 1 - (distances[cluster] / np.max(distances))
        return cluster, confidence
