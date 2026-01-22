import requests
import json
import time
import streamlit as st

from pathlib import Path

JSON_FILE = Path("data/document.json")
FETCH_URL = "https://newsapi.org/v2/everything"
API_KEY = "cc28d4855f404ad582cd1dbb20fcda41"


def fetch_category_data(category: str, source: str, count: int = 50) -> list:

    params = {
        # "q": category,
        "sources": source,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apikey": API_KEY,
    }

    if source == "bbc-news":
        params = {**params, "q": category}

    articles = []
    page = 1
    while len(articles) < count:
        params["page"] = page
        resp = requests.get(FETCH_URL, params=params)
        data = resp.json()

        if data["status"] != "ok" or not data["articles"]:
            break

        for article in data["articles"][: count - len(articles)]:
            if article["title"] and len(article["title"]) > 20:
                articles.append(
                    {
                        "id": len(articles),
                        "text": article["title"]
                        + ". "
                        + (article["description"] or ""),
                        "category": category,
                        "url": article["url"],
                        "source": article["source"]["name"],
                    }
                )

        page += 1
        time.sleep(0.2)

    return articles[:count]


def collect():

    all_docs = []

    with st.status("Loading file ...") as status:
        category_mapping = {
            "business": "bloomberg,the-wall-street-journal",
            "entertainment": "buzzfeed, mashable,ign",
            "health": "medical-news-today, bbc-news",
        }

        for category, sources in category_mapping.items():
            for source in sources.split(","):
                status.update(label=f"Fetching from {source} ...")
                print(f"Fetching from {source}")
                docs = fetch_category_data(category, source)
                all_docs.extend(docs)
                time.sleep(1)

        # SAVE
        JSON_FILE.parent.mkdir(exist_ok=True)
        with open(JSON_FILE, "w") as f:
            status.update(label="Dumping data to json file ...")
            json.dump(all_docs, f, indent=2)

        print(f"Saved {len(all_docs)} articles to {JSON_FILE}")
        status.update(
            label="Saved articles to file. Reload to proceed!",
            state="complete",
            expanded=False,
        )
        st.stop()
