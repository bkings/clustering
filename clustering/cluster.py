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

    k = 6
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def __init__(self):
        self.docs = []

    def label_clusters(_self, kmeans, vectorizer, docs, df):
        cluster_profiles = {}

        for cluster_id in range(Cluster.k):
            cluster_df = df[df["cluster"] == cluster_id]
            if cluster_df.empty:
                continue

            # 2. Get corresponding doc texts (df index = docs index)
            cluster_indices = cluster_df.index.tolist()  # [5, 12, 23, ...]
            cluster_texts = [_self.preprocess(docs[i]["text"]) for i in cluster_indices]

            # 3. Top TF-IDF words for this cluster
            if cluster_texts:
                vec_cluster = vectorizer.transform(cluster_texts)
                feature_names = vectorizer.get_feature_names_out()
                top_indices = vec_cluster.sum(axis=0).argsort()[0, -20:][::-1]
                top_words = [feature_names[i] for i in top_indices]
            else:
                top_words = []

            # 4. Majority category
            majority_cat = (
                cluster_df["category"].mode().iloc[0]
                if not cluster_df["category"].mode().empty
                else "Unknown"
            )

            # 5. Sample doc (first in cluster)
            sample_idx = cluster_indices[0]
            sample_doc = docs[sample_idx]["text"][:150] + "..."

            cluster_profiles[cluster_id] = {
                "cluster_id": cluster_id,
                "label": f"{majority_cat} ({', '.join(str(item) for item in top_words[:5])})",
                "top_words": top_words,
                "majority_category": majority_cat,
                "sample_doc": sample_doc,
                "doc_count": len(cluster_indices),
            }

        return cluster_profiles

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

    # @st.cache_data
    def preprocess(_self, text):
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()
        tokens = [
            Cluster.stemmer.stem(t)
            for t in tokens
            if t not in Cluster.STOP_WORDS and len(t) > 2
        ]
        return " ".join(tokens)

    # @st.cache_data
    def train_model(_self, docs):
        texts = [_self.preprocess(d["text"]) for d in docs]
        vectorizer = TfidfVectorizer(
            max_features=5000, max_df=0.8, min_df=2, ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=Cluster.k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        print(np.bincount(clusters))
        print(clusters)

        df = pd.DataFrame(
            {
                "text": [d["text"][:150] + "..." for d in docs],
                "category": [d["category"] for d in docs],
                "cluster": clusters,
            }
        )
        return kmeans, vectorizer, X, df

    def predict_cluster(self, query: str, kmeans, vectorizer: TfidfVectorizer):
        query = self.preprocess(query)
        vec = vectorizer.transform([query])
        distances = kmeans.transform(vec)[0]
        print(distances)
        cluster = kmeans.predict(vec)[0]
        confidence = 1 - (distances[cluster] / np.max(distances))
        return cluster, confidence
