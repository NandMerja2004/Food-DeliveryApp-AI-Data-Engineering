import json
import os
import re
from pathlib import Path

import ollama
import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "gemma4:e2b"

FORBIDDEN_WORDS = [
    "drop",
    "delete",
    "truncate",
    "alter",
    "update",
    "insert",
    "create",
    "replace",
    "grant",
    "revoke",
]

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method",
]

SCHEMA = """
Tables available (Snowflake). Use bare table names, no database or schema prefix.

FCT_ORDERS(order_id, order_date, customer_id, restaurant_id, city, cuisine,
           payment_method, order_status, is_delivered, sales_amount, discount,
           delivery_fee, gst, customer_rating, delivery_time_min)
DIM_RESTAURANT(restaurant_id, restaurant_name, city, cuisine, rating, cost_for_two)
DIM_CUSTOMER(customer_id, customer_name, age, age_segment, gender, city)
MART_DAILY_CITY_REVENUE(order_date, city, orders, cancel_rate, gmv, aov)
MART_RESTAURANT_PERFORMANCE(restaurant_id, restaurant_name, city, cuisine,
                            orders, revenue, avg_customer_rating, cancel_rate)
MART_DELIVERY_SLA(city, order_hour, delivered_orders, p50_delivery_min, late_rate)

Note: gmv means delivered revenue. Prefer the MART_ tables when they fit the question. MART_DAILY_CITY_REVENUE has one row per city per day, so use SUM(gmv) and GROUP BY city for total city GMV.
"""

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert. Write ONE SELECT query that answers the question.

Rules:
- SELECT queries only, never modify data.
- Use bare table names (FCT_ORDERS, not ZOMATO.MARTS.FCT_ORDERS).
- STRICTLY use only columns that exist in the table(s) you select from. Double-check every column against the schema below.
- Only use a MART_ table if EVERY column needed for the question exists in that MART_ table. Otherwise, use FCT_ORDERS (for example, payment_method and cuisine cancel rates are in FCT_ORDERS, not MART_DAILY_CITY_REVENUE).
- To calculate cancel rate from FCT_ORDERS, use: AVG(CASE WHEN LOWER(order_status) LIKE '%cancel%' THEN 1.0 ELSE 0.0 END).
- Add a LIMIT of 100 or less, unless the question asks for a single total.
- Reply as JSON in this exact format: {{"sql": "your query here"}}

{SCHEMA}
"""


@st.cache_resource
def get_connection():
    conn_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "ZOMATO"),
        "schema": "MARTS",
    }
    role = os.getenv("SNOWFLAKE_ROLE")
    if role:
        conn_params["role"] = role

    return snowflake.connector.connect(**conn_params)


def generate_sql(
    question: str, previous_sql: str = None, error_msg: str = None
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if previous_sql and error_msg:
        messages.append(
            {"role": "assistant", "content": json.dumps({"sql": previous_sql})}
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That SQL failed in Snowflake with this error:\n{error_msg}\n"
                    "Fix the query using ONLY valid columns from the schema and return JSON."
                ),
            }
        )

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        format="json",
        options={"temperature": 0},
        think=False,
    )
    sql = json.loads(response.message.content)["sql"]

    sql = re.sub(r"(?i)\bZOMATO\.MARTS\.", "", sql)
    sql = re.sub(r"(?i)\b(?:ZOMATO|MARTS)\.", "", sql)
    return sql.strip().rstrip(";")


def is_safe(sql: str) -> bool:
    lowered = sql.lower().strip()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False

    if ";" in lowered:
        return False

    for word in FORBIDDEN_WORDS:
        if re.search(rf"\b{word}\b", lowered):
            return False

    return True


def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        return cursor.execute(sql).fetch_pandas_all()


st.set_page_config(page_title="Chat with your Zomato Data")
st.title("Chat with your Zomato Data")
st.caption(f"Ask in English, {MODEL} writes the SQL, Snowflake runs it")

with st.sidebar:
    st.header("Example Questions")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

question = st.text_input(
    "Enter your question here",
    placeholder="e.g. Top 10 restaurants by revenue in Bangalore",
)

if question:
    with st.spinner("Generating SQL..."):
        try:
            sql = generate_sql(question)
        except Exception as e:
            st.error(f"Error generating SQL: {e}")
            st.stop()

    st.code(sql, language="sql")

    if not is_safe(sql):
        st.error("The generated SQL is not safe to run. Please modify your question.")
    else:
        try:
            with st.spinner("Running query on Snowflake..."):
                df = run_query(sql)
        except Exception as first_err:
            with st.spinner("SQL had an error, asking model to fix it..."):
                try:
                    sql = generate_sql(
                        question, previous_sql=sql, error_msg=str(first_err)
                    )
                except Exception as e:
                    st.error(f"Error regenerating SQL: {e}")
                    st.stop()

            st.code(sql, language="sql")

            if not is_safe(sql):
                st.error("The revised SQL is not safe to run.")
                st.stop()

            try:
                with st.spinner("Running revised query on Snowflake..."):
                    df = run_query(sql)
            except Exception as e:
                st.error(f"Error running query: {e}")
                st.stop()

        st.success(f"{len(df)} rows returned")
        st.dataframe(df, hide_index=True)

        if (
            not df.empty
            and len(df.columns) == 2
            and pd.api.types.is_numeric_dtype(df.iloc[:, 1])
        ):
            st.bar_chart(df, x=df.columns[0], y=df.columns[1])