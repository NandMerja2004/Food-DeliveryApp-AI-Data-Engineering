import json
import os
from pathlib import Path

import ollama
import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "gemma4:e2b"
SAMPLE_N = 5

TOPICS = [
    "food_quality",
    "delivery_speed",
    "packaging",
    "pricing",
    "portion_size",
    "customer_service",
    "other",
]


def get_connection():
    conn_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "ZOMATO"),
    }
    role = os.getenv("SNOWFLAKE_ROLE")
    if role:
        conn_params["role"] = role

    return snowflake.connector.connect(**conn_params)


def get_review_columns(cur) -> tuple[str, str]:
    cur.execute("DESC TABLE ZOMATO.RAW.REVIEWS")
    cols = [row[0].upper() for row in cur.fetchall()]

    id_col = "REVIEW_ID" if "REVIEW_ID" in cols else cols[0]
    text_candidates = [
        "REVIEW_TEXT",
        "COMMENT",
        "COMMENTS",
        "REVIEW",
        "TEXT",
        "FEEDBACK",
    ]
    text_col = next((c for c in text_candidates if c in cols), None)

    if not text_col:
        raise ValueError(
            f"Could not find a review text column in ZOMATO.RAW.REVIEWS. Found: {cols}"
        )

    return id_col, text_col


def analyze_review(review_text: str) -> dict:
    prompt = f"""
Analyze this food delivery review and reply ONLY as valid JSON with these exact keys:
- "topic": one of {TOPICS}
- "sentiment_label": one of "positive", "neutral", "negative"
- "sentiment_score": number between -1.0 (very negative) and 1.0 (very positive)
- "key_issue": short phrase describing the main complaint, or null if there is no complaint

Review: {review_text}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
        think=False,
    )

    data = json.loads(response.message.content)

    topic = str(data.get("topic", "other")).strip().lower()
    if topic not in TOPICS:
        topic = "other"

    sentiment_label = str(data.get("sentiment_label", "neutral")).strip().lower()
    if sentiment_label not in {"positive", "neutral", "negative"}:
        sentiment_label = "neutral"

    try:
        sentiment_score = float(data.get("sentiment_score", 0.0))
    except (TypeError, ValueError):
        sentiment_score = 0.0
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    raw_issue = data.get("key_issue")
    if raw_issue is None or str(raw_issue).strip().lower() in {
        "",
        "null",
        "none",
        "n/a",
        "no issue",
        "no complaint",
    }:
        key_issue = None
    else:
        key_issue = str(raw_issue).strip()

    return {
        "topic": topic,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "key_issue": key_issue,
    }


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS ZOMATO.AI")
            cur.execute(
                """
                CREATE OR REPLACE TABLE ZOMATO.AI.REVIEW_ENRICHED (
                    REVIEW_ID VARCHAR,
                    TOPIC VARCHAR,
                    SENTIMENT_LABEL VARCHAR,
                    SENTIMENT_SCORE FLOAT,
                    KEY_ISSUE VARCHAR,
                    ENRICHED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
                """
            )

            id_col, text_col = get_review_columns(cur)
            cur.execute(
                f"""
                SELECT {id_col}, {text_col}
                FROM ZOMATO.RAW.REVIEWS
                WHERE {text_col} IS NOT NULL
                LIMIT {SAMPLE_N}
                """
            )
            rows = cur.fetchall()

            saved = 0
            for review_id, review_text in rows:
                try:
                    result = analyze_review(str(review_text))
                    cur.execute(
                        """
                        INSERT INTO ZOMATO.AI.REVIEW_ENRICHED
                            (REVIEW_ID, TOPIC, SENTIMENT_LABEL, SENTIMENT_SCORE, KEY_ISSUE)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            str(review_id),
                            result["topic"],
                            result["sentiment_label"],
                            result["sentiment_score"],
                            result["key_issue"],
                        ),
                    )
                    saved += 1
                    print(
                        f"Enriched {review_id}: "
                        f"{result['topic']} | {result['sentiment_label']} ({result['sentiment_score']})"
                    )
                except Exception as e:
                    print(f"Skipping review {review_id} due to error: {e}")

            conn.commit()
            print(f"Done. Saved {saved} enriched reviews to ZOMATO.AI.REVIEW_ENRICHED.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()