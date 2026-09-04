# 🛡️ SecureRAG: Zero-Dependency RAG Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-orange.svg)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📌 Overview
SecureRAG is a lightweight, fully custom Retrieval-Augmented Generation (RAG) pipeline built from scratch. It ingests local documents, chunks and embeds the text, indexes it in an in-memory vector database, and grounds Large Language Model (LLM) responses in the retrieved context while strictly mitigating indirect prompt injection vulnerabilities.

---

## 🏗️ Architecture Flow

```text
[Document: PDF/TXT] 
       │
       ▼
 [Text Extraction] ───► [Chunking (Sliding Window)]
                               │
                               ▼
                        [Embeddings: all-MiniLM-L6-v2]
                               │
                               ▼
[User Query] ─────────► [FAISS Vector Store]
                               │
                               ▼
                       [Top-K Retrieval]
                               │
                               ▼
                    [Security Hardened Prompt]
                               │
                               ▼
                     [LLM: openai/gpt-oss-20b]
                               │
                               ▼
                      [Grounded Answer]
```

---

## 🛠️ Tech Stack & Design Justifications

* **LLM Provider:** `openai/gpt-oss-20b` (via **Groq**). This is a 20-billion parameter open-weight model optimized for low-latency inference on Groq's hardware. It provides the necessary reasoning capabilities while keeping API responses nearly instantaneous.
* **Vector Database:** **FAISS** (Facebook AI Similarity Search). `faiss-cpu` was chosen because it runs completely in-memory using an exact `IndexFlatL2` search, eliminating the overhead and security risks of persistent local vector DB servers.
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`. This model maps sentences and paragraphs into a dense 384-dimensional vector space. It was selected because it runs completely locally on CPU without requiring an external API, ensuring fast and free indexing.
* **Document Loaders:** `PyPDF2` and native Python text processing. Kept lightweight to avoid the heavy dependencies of frameworks like LangChain or LlamaIndex.

---

## 📂 Repository Structure

```text
├── data/
│   ├── sample.pdf          # Add your PDFs here
│   └── sample.txt          # Add your text/markdown files here
├── .env.example            # Template for environment variables
├── .gitignore              # Ignores sensitive data and virtual environments
├── rag.py                  # Main execution script containing the full pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Setup & Run Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up your API Key

You can either create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_actual_api_key_here
```

*Note: If the script cannot detect your `.env` file (e.g., due to Windows/OneDrive syncing quirks), the pipeline will safely pause in the terminal and prompt you to paste your key securely using Python's `getpass` module.*

### 4. Run the Pipeline

Place a test document in the `data/` folder and execute the script:

```bash
python rag.py
```

---

## 💡 Key Features & Capabilities

* **Custom Chunking Strategy:** Implements a word-count sliding window (150 words per chunk with a 30-word overlap) to ensure semantic continuity across split boundaries without truncating words in half.
* **Prompt-Injection Defense:** Retrieved contexts are isolated within strict XML tags (`<context>...</context>`). The system prompt explicitly commands the LLM to treat the context as passive data and ignore any adversarial directives injected into the source documents.
* **Memory-Safe Vector Operations:** Embeddings are kept entirely in RAM during execution. No unencrypted `.pkl` files or raw vectors are written to disk, preventing embedding inversion attacks and unsafe deserialization.
* **Interactive Chat Loop:** Allows continuous Q&A with the ingested document until the user exits the program.

---

## 💻 Example Usage

```text
--- 1. Loading Document ---

--- 2. Chunking Text ---
Created 42 chunks.

--- 3. Indexing Chunks in FAISS ---
Embedding 42 chunks using SentenceTransformer(name_or_path='all-MiniLM-L6-v2')...
FAISS Index created successfully.

Ask a question (or type 'quit' to exit): What is the purpose of FAISS?

[Thinking... Retrieving context from document...]
  -> Found relevant chunk 1
  -> Found relevant chunk 2
  -> Found relevant chunk 3

🤖 [Answer]:
Based on the provided context, FAISS was developed by Facebook AI Research to enable efficient similarity search for vector embeddings.
--------------------------------------------------
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).
