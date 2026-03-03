# Swiggy Annual Report RAG System

A production-quality Retrieval Augmented Generation (RAG) system built in Python, designed to answer questions **strictly** from the Swiggy Annual Report PDF. It utilizes LangChain, FAISS, and Google's Gemini models (`models/text-embedding-004` and `gemini-1.5-flash`) to provide highly accurate, hallucination-free answers with precise page citations.

## Key Features
- **Strict Anti-Hallucination:** System prompt forces `gpt-4o-mini` to answer *only* from ingested chunks.
- **Source Citations:** Every response includes metadata pointing to exact page numbers and quotes.
- **Modular Pipeline:** Clean separation of configuration, document processing, embedding storage, and API logic.
- **Background Ingestion:** FastAPI handles multi-megabyte PDF chunking and FAISS indexing via a background task.
- **Evaluation Module:** Includes a robust assessment script using `ragas` for Precision, Recall, Faithfulness, and Relevancy.

---

## 🛠️ Setup & Installation

**1. Create a virtual environment (Optional but Recommended)**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Copy `.env.example` to `.env` and add your Gemini Key:
```bash
GEMINI_API_KEY=AIza...
```

**4. Add the Data**
Place the Swiggy Annual Report PDF inside the `data/` directory named exactly:
`data/swiggy_annual_report.pdf`
*(Create the directory if it does not exist).*

---

## 🚀 Usage Guide

### Option 1: CLI Interface
The simplest way to use the system locally is via the command-line interface.

**Step 1: Ingest the Document**
This builds the local FAISS index. You only need to run this once.
```bash
python main.py --ingest
```

**Step 2: Start Interactive Q&A**
```bash
python main.py
```
*Example Interactions:*
```
Query: What was the total revenue in 2023?
---------------------------------------------------------
ANSWER:
The total revenue for 2023 was X million USD.

CONFIDENCE: High

SOURCES:
  [1] Page 42: "The company reported a total gross revenue of X million USD in 2023."
---------------------------------------------------------
```

### Option 2: FastAPI Server
To serve requests via HTTP endpoints (e.g., for front-end integration).

**Start the Server:**
```bash
uvicorn src.api:app --reload
```

**Access Endpoints:**
- Swagger UI / Docs: `http://localhost:8000/docs`
- `POST /ingest` (Run first to build index)
- `POST /query`
   ```json
   {
       "question": "How many delivery partners are active?"
   }
   ```

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Swiggy Report PDF] --> B[PyPDFLoader]
    B --> C[RecursiveTextSplitter]
    C -->|Docs + Page Metadata| D[Gemini Embeddings]
    D --> E[(FAISS Vector Store)]
    
    F[User Query] --> G[Retrieval Top-K]
    E --> G
    G --> H[Strict System Prompt]
    H --> I[Gemini 1.5 Flash Generator]
    I --> J{Validation}
    J -->|Answer Found| K[Structured Output with Citations]
    J -->|Not in Context| L[Return: "Info Not Available"]
```

## 🧪 Evaluation

Run the Ragas pipeline generator to measure `Faithfulness`, `Answer Relevancy`, `Context Recall`, and `Context Precision`.
*(Ensure the data/swiggy_annual_report.pdf has been ingested first)*.

```bash
python -m src.evaluate
```
*Note: Update the ground truths inside `src/evaluate.py` to match the real contents of your supplied PDF before running.*
