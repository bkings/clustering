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
cluster_profiles = cluster.label_clusters(kmeans, vectorizer, docs, df)

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
        profile = cluster_profiles[pred_cluster]
        st.success(
            f"**Predicted Cluster: ({profile['cluster_id']}) {profile["majority_category"]}** (Confidence: {conf:.1%} | Docs: {profile['doc_count']})"
        )

        with st.expander("Cluster Profile"):
            st.write(
                f"**Top Keywords**: {', '.join(str(item) for item in profile['top_words'][:12])}"
            )
            st.write(f"**Majority Category**: {profile['majority_category']}")
            st.write("**Sample Doc**:")
            st.caption(profile["sample_doc"])

        # Robustness test cases
        tests = {
            "Entertainment": "Animation labs creation",
            "Health": "New vaccine reduces flu that are on rise",
            "Short": "Blood pressure levels",
            "Business": "Silver and gold prices hike",
            "Mixed": "Movie profits boost overall financial state",
            "Specific long query with stop words": "latest movie reviews that were above average and most people enjoyed watching today but did not yesterday",
        }
        st.subheader("Test Cases")
        for label, test in tests.items():
            pred, conf = cluster.predict_cluster(test, kmeans, vectorizer)
            clus_profile = cluster_profiles[pred]
            st.write(
                f"**{label}**: {test} - Cluster {pred} '{clus_profile['majority_category']}'"
            )

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
    for cluster in sorted(set(df["cluster"])):
        st.subheader(f"Cluster {cluster}")
        with st.container(height=300):
            cluster_docs = df[df["cluster"] == cluster].head(30)
            for _, row in cluster_docs.iterrows():
                st.write(f"**{row['category']}**: {row['text']}")

st.markdown("---")
