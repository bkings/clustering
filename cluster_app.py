import streamlit as st
import pandas as pd
import plotly.express as px
from doc_gathering.collector import collect
from clustering.cluster import Cluster
from sklearn.decomposition import PCA

st.set_page_config(page_title="Document Clustering", layout="wide")
st.title("Clustering by cagetory")
st.markdown("K-Means clustering (Business/Entertainment/Health)")

cluster = Cluster()
try:
    docs = cluster.load_file()
    load_docs = False
except:
    load_docs = st.button("Load Documents")
    if load_docs:
        collect()
    st.stop()


kmeans, vectorizer, X, df = cluster.train_model(docs)

tab1, tab2, tab3 = st.tabs(["Cluster New Document", "Results", "Sample Docs"])

with tab1:
    st.header("Predict Cluster")

    queryCol, buttonCol = st.columns([15, 1], gap=None)

    with queryCol:
        new_document = st.text_input(
            "Enter query:",
            placeholder="Eg. 'Top business ...' or 'Latest celebrity gossip ...'",
            label_visibility="collapsed",
        )

    with buttonCol:
        search_clicked = st.button("🔍")

    if new_document or (new_document and search_clicked):
        pred_cluster, conf = cluster.predict_cluster(
            new_document.strip(), kmeans, vectorizer
        )
        st.success(f"**Predicted Cluster: {pred_cluster}** (Confidence: {conf:.2f})")

        # Robustness test cases
        tests = {
            "Entertainment": "Oscars 2026 best picture nominees announced.",
            "Health": "New vaccine reduces flu cases by 70%.",
            "Hard Short": "Flu shot",
            "Hard No Stop": "the a vaccine works great",
            "Mixed": "Movie profits boost UK economy",
            "Business": "Elon musk becomes the most richest businessman in the world",
        }
        st.subheader("Test Cases")
        for label, test in tests.items():
            pred, conf = cluster.predict_cluster(test, kmeans, vectorizer)
            st.write(f"**{label}**: {test} - Cluster {pred} ({conf:.2f})")

with tab2:
    st.header("Clustering quality")
    crosstab = pd.crosstab(df["category"], df["cluster"])
    fig = px.imshow(
        crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        color_continuous_scale="Blues",
        title="Ground Truth vs Predicted Clusters",
    )
    st.plotly_chart(fig)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    df_viz = df.copy()
    df_viz["pca_x"] = X_pca[:, 0]
    df_viz["pca_y"] = X_pca[:, 1]
    fig2 = px.scatter(
        df_viz,
        x="pca_x",
        y="pca_y",
        color="cluster",
        hover_data=["category", "text"],
        title="PCA: Clusters Separation",
    )
    st.plotly_chart(fig2)

with tab3:
    st.header("Sample documents")
    for cluster in df["cluster"].unique():
        st.subheader(f"Cluster {cluster}")
        cluster_docs = df[df["cluster"] == cluster].head(3)
        for _, row in cluster_docs.iterrows():
            st.write(f"**{row['category']}**: {row['text']}")

st.markdown("---")