import os
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "gemma4:e2b"
NUM_REVIEWS = 500
TOP_K = 5
CACHE_FILE = Path(__file__).resolve().parent / "review_embeddings.parquet"


def get_snowflake_conn():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE", "ZOMATO"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


def embed_texts(texts, batch_size: int = 50):
    if isinstance(texts, str):
        return ollama.embed(model=EMBED_MODEL, input=texts).embeddings[0]

    all_embeddings = []
    progress = st.progress(0, text="Embedding reviews locally (one-time setup)...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = ollama.embed(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(res.embeddings)

        done = min(i + batch_size, len(texts))
        progress.progress(
            done / len(texts),
            text=f"Embedded {done}/{len(texts)} reviews...",
        )

    progress.empty()
    return all_embeddings


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)

    conn = get_snowflake_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT CITY, RATING, COMMENT
            FROM ZOMATO.STAGING.STG_REVIEWS
            WHERE COMMENT IS NOT NULL
              AND TRIM(COMMENT) != ''
            LIMIT {NUM_REVIEWS}
            """
        )
        cols = [col[0].lower() for col in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()
        conn.close()

    df["embedding"] = embed_texts(df["comment"].astype(str).tolist())
    df.to_parquet(CACHE_FILE, index=False)
    return df


def find_similar_reviews(
    question: str, review_df: pd.DataFrame, top_k: int = TOP_K
) -> pd.DataFrame:
    query_vec = np.array(embed_texts(question), dtype=np.float32)
    matrix = np.vstack(review_df["embedding"].values).astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    scores = np.dot(matrix, query_vec) / np.where(norms == 0, 1e-10, norms)

    top_idx = np.argsort(scores)[::-1][:top_k]
    return review_df.iloc[top_idx].reset_index(drop=True)


def ask_llm(question: str, top_reviews: pd.DataFrame) -> str:
    context = "\n".join(
        f"- [{row.city} | Rating: {row.rating}] {row.comment}"
        for row in top_reviews.itertuples()
    )

    res = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided Zomato reviews. "
                    "Be concise, specific, and factual. If the reviews do not contain "
                    "enough information, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Reviews:\n{context}\n\nQuestion: {question}",
            },
        ],
        think=False,
    )
    return res.message.content


st.set_page_config(page_title="Zomato Review Chat")
st.title("Zomato Review Chat")
st.caption(f"Searching {NUM_REVIEWS} reviews, answering with {CHAT_MODEL}")

review_df = load_reviews()

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder="e.g. What are the most common complaints about delivery?",
)

if question:
    with st.spinner("Searching reviews and generating answer..."):
        top_reviews = find_similar_reviews(question, review_df)
        answer = ask_llm(question, top_reviews)

    st.markdown("**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(top_reviews[["city", "rating", "comment"]], hide_index=True)