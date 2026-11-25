# Food Delivery AI & Data Engineering Platform

An end-to-end modern data stack and applied AI platform for food delivery analytics. The pipeline orchestrates raw data ingestion, analytical modeling in **Snowflake** with **dbt**, automated local LLM review enrichment using **Ollama (`gemma4:e2b`)** inside **Apache Airflow 3**, and interactive **Streamlit** interfaces for **Text-to-SQL** and **RAG** chat.

---

## Architecture

```text
Raw Food Delivery Data (CSV / Stage)
               │
               ▼
   ┌───────────────────────┐
   │   Snowflake (RAW)     │
   └───────────┬───────────┘
               │
               ▼  Orchestrated by Apache Airflow 3 (Docker)
   ┌───────────────────────┐        ┌─────────────────────────────────┐
   │  dbt Core (Staging &  │        │  ai/enrich_reviews.py           │
   │  Dimensional Marts)   │        │  Local Ollama (gemma4:e2b)      │
   └───────────┬───────────┘        │  → Topic, Sentiment, Key Issue  │
               │                    └────────────────┬────────────────┘
               │                                     ▼
               │                    ┌─────────────────────────────────┐
               │                    │ Snowflake (AI.REVIEW_ENRICHED)  │
               │                    └────────────────┬────────────────┘
               ▼                                     ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │             dbt AI Analytics Layer (mart_review_insights)        │
   └───────────────────────────────┬──────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
      ┌────────────────────┐             ┌──────────────────────┐
      │ Streamlit Text-to- │             │ Streamlit RAG Chat   │
      │ SQL (text_to_sql)  │             │ (rag_chat.py)        │
      └────────────────────┘             └──────────────────────┘
```

---

## Key Features

- **Containerized Airflow 3 Orchestration:** Multi-container Apache Airflow (`3.0.3`) running `LocalExecutor` with dedicated `apiserver`, `scheduler`, `dag-processor`, and PostgreSQL metadata services.
- **Isolated dbt Transformation Layer:** Runs `dbt-snowflake==1.8.*` in a dedicated container virtual environment (`/opt/airflow/dbt_venv`) to prevent dependency conflicts with Airflow providers.
- **Zero-Cost Local LLM Enrichment:** Connects Dockerized Airflow tasks to a host Ollama instance (`gemma4:e2b`) to classify customer reviews into structured JSON fields (`topic`, `sentiment_label`, `sentiment_score`, `key_issue`) and load results into Snowflake.
- **AI-Ready Data Marts:** Joins LLM-enriched review metadata with core warehouse models (`dim_customer`, `fct_order_items`, `mart_review_insights`) for operational reporting.
- **Conversational Analytics (Streamlit):**
  - **Text-to-SQL (`ai/text_to_sql.py`):** Translates natural language business questions into executable Snowflake SQL queries.
  - **Review RAG Chat (`ai/rag_chat.py`):** Performs semantic search over customer review embeddings to surface recurring complaints and trends.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Cloud Data Warehouse** | Snowflake |
| **Transformation & Modeling** | dbt Core (`dbt-snowflake==1.8.*`) |
| **Orchestration** | Apache Airflow 3.0.3 (`LocalExecutor`, FAB Auth) |
| **Containerization** | Docker & Docker Compose |
| **Local LLM Inference** | Ollama (`gemma4:e2b`) |
| **Analytics & AI UI** | Streamlit |
| **Language** | Python 3.12, SQL |

---

## Project Structure

```text
.
├── ai/
│   ├── enrich_reviews.py         # Local LLM review enrichment job → Snowflake AI schema
│   ├── rag_chat.py               # Streamlit RAG chatbot for customer review analysis
│   └── text_to_sql.py            # Streamlit Text-to-SQL analytics copilot
├── airflow/
│   ├── dags/
│   │   └── zomato_batch.py       # End-to-end Airflow DAG (ELT + LLM + dbt AI build)
│   ├── Dockerfile                # Airflow 3.0.3 image + Snowflake/FAB providers + dbt venv + Ollama
│   └── docker-compose.yaml       # Multi-container Airflow 3 stack with host Ollama networking
├── zomato/                       # dbt project
│   ├── macros/
│   │   └── generate_schema_name.sql
│   ├── models/
│   │   ├── staging/              # Cleaned source views/tables
│   │   └── marts/                # dim_customer, fct_order_items, mart_review_insights
│   └── dbt_project.yml
└── .gitignore                    # Excludes credentials (.env, profiles.yml), logs, parquet & raw data
```

---

## Getting Started

### Prerequisites

- **Docker Desktop** (running)
- **Snowflake** account with database (`ZOMATO`) and warehouse (`ZOMATO_WH`)
- **Ollama** installed locally with the model pulled:
  ```bash
  ollama pull gemma4:e2b
  ```
  > **Docker Networking Note:** In Ollama Desktop settings, enable **Expose Ollama to the network** so Docker containers can reach the host via `http://host.docker.internal:11434`.

### 1. Clone the Repository

```bash
git clone [https://github.com/NandMerja2004/Food-DeliveryApp-AI-Data-Engineering.git](https://github.com/NandMerja2004/Food-DeliveryApp-AI-Data-Engineering.git)
cd Food-DeliveryApp-AI-Data-Engineering
```

### 2. Configure Credentials

Create environment files (excluded from Git via `.gitignore`):

**`airflow/.env` and `ai/.env`:**
```env
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=ZOMATO_WH
SNOWFLAKE_DATABASE=ZOMATO
SNOWFLAKE_ROLE=your_role
```

**`zomato/profiles.yml`:**
Configure your dbt Snowflake profile referencing the environment variables above (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`).

### 3. Build and Start Apache Airflow

```bash
cd airflow
docker-compose up -d --build
```

- Open the Airflow UI at **`http://localhost:8080`** (default login: `admin` / `admin`).
- Enable and trigger the **`zomato_batch`** DAG to run ingestion, dbt core models, `enrich_reviews`, and `dbt_build_ai`.

### 4. Launch the Streamlit AI Apps

From the project root (with Python dependencies `streamlit`, `ollama`, `snowflake-connector-python`, and `python-dotenv` installed):

```bash
# Natural language to Snowflake SQL
streamlit run ai/text_to_sql.py

# Semantic RAG chat over customer reviews
streamlit run ai/rag_chat.py
```

---

## LLM Enrichment Schema (`ZOMATO.AI.REVIEW_ENRICHED`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `REVIEW_ID` | `VARCHAR` | Unique identifier from `ZOMATO.RAW.REVIEWS` |
| `TOPIC` | `VARCHAR` | Classified category (`food_quality`, `delivery_speed`, `packaging`, `pricing`, `portion_size`, `customer_service`, `other`) |
| `SENTIMENT_LABEL` | `VARCHAR` | `positive`, `neutral`, or `negative` |
| `SENTIMENT_SCORE` | `FLOAT` | Continuous score from `-1.0` to `1.0` |
| `KEY_ISSUE` | `VARCHAR` | Concise summary of the primary complaint (`NULL` if none) |
| `ENRICHED_AT` | `TIMESTAMP_NTZ` | Timestamp of LLM processing |

---

## Security & Hygiene

Credentials (`.env`, `profiles.yml`), local embedding caches (`*.parquet`), execution logs, dbt build artifacts (`target/`), and raw datasets (`data/`) are strictly ignored via `.gitignore`.
