Project Report Analyzer (RAG-based Q&A System)
Overview

The Project Report Analyzer is a Retrieval-Augmented Generation (RAG) application designed to analyze complex, multi-page project reports in PDF format and answer user queries with accurate, source-attributed responses.

The system supports querying across single or multiple project reports, handling semi-structured content such as tables, timelines, budgets, and narrative sections. It is built with a production-oriented architecture, emphasizing explainability, modularity, and extensibility.

Key Features

📄 Multi-PDF Upload
Upload one or more project reports in PDF format within a single session.

🧠 Structure-Aware RAG Pipeline
Handles mixed document structures (tables, key-value fields, narrative text).

🔍 Semantic Retrieval Across Documents
Queries can retrieve information from one or multiple reports seamlessly.

🧾 Source-Attributed Answers
Every answer includes document name and page-level citations.

💬 Chat-Style Interface
Conversational querying with session persistence.

🚀 Streaming-Ready Generation
Backend supports token streaming from the LLM.

🐳 Dockerized Deployment
Fully containerized for reproducibility and portability.

High-Level Architecture
User (Chat UI)
      |
      v
Streamlit Interface
      |
      v
RAG Pipeline (LangGraph)
 ├── Retrieval (FAISS)
 ├── Context Assembly
 ├── LLM Generation (LM Studio)
 └── Citation Mapping
      |
      v
Answer + Sources

Technology Stack
Backend & Core Logic

Python 3.10

LangChain – LLM orchestration primitives

LangGraph – Graph-based RAG orchestration

FAISS – Vector similarity search

Unstructured – Layout-aware PDF parsing

LLM & Embeddings

LM Studio (OpenAI-compatible API)

Local LLM (e.g., llama-2-7b-chat)

Hugging Face / OpenAI-compatible embeddings

Frontend

Streamlit – Interactive chat UI

Infrastructure

Docker – Containerized deployment

Git + Semantic Versioning – Source control and releases

Document Processing Pipeline
1. PDF Ingestion

PDFs are uploaded via the Streamlit UI.

Parsing is performed using Unstructured with a layout-aware strategy.

Works on Windows without native dependencies (Poppler avoided).

2. Chunking Strategy

Type-aware chunking is applied:

Tables preserved as atomic chunks to retain row relationships.

Narrative text split using overlap-aware recursive chunking.

Titles and headers associated with subsequent content.

Each chunk retains metadata:

Source document

Page number(s)

Chunk type

3. Embedding Generation

Chunks are converted into vector embeddings using a consistent embedding model.

The same embedding space is used for both documents and queries to ensure alignment.

Vector Store & Retrieval

FAISS is used as the vector store.

All document chunks from all uploaded PDFs are indexed together.

Two retrieval modes:

Similarity Search for direct factual queries

MMR (Maximal Marginal Relevance) for comparative or cross-document queries

Retrieved chunks are passed forward with full metadata for explainability.

RAG Orchestration & Answer Generation

Implemented using LangGraph for explicit, node-based control.

Pipeline stages:

Query intake

Semantic retrieval

Context assembly

LLM generation

Citation construction

Prompting strategy strictly constrains the model to:

Use only retrieved context

Avoid hallucinations

Explicitly indicate when information is missing

Source Attribution & Explainability

Each answer includes:

Source document name

Page number(s)

Citations are derived directly from retrieved chunk metadata.

This enables:

Transparent AI outputs

Auditable responses

Trustworthy comparison across multiple reports

User Interface

Chat-based interaction using Streamlit’s conversational components.

Features:

Multi-document upload

Persistent session state

Follow-up questions without re-upload

Clear separation of answer and sources

Project Structure
project-report-analyzer/
│
├── app/
│   ├── ingestion/        # PDF loading & parsing
│   ├── chunking/         # Structure-aware chunking
│   ├── embeddings/       # Embedding provider
│   ├── vectorstore/      # FAISS abstraction
│   ├── rag/              # LangGraph RAG pipeline
│   ├── ui/               # Streamlit chat UI
│   └── core/             # Logging, config, exceptions
│
├── notebooks/            # Experiments / prototyping
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md

Running the Application
Local (Without Docker)

Create virtual environment and install dependencies:

pip install -r requirements.txt


Set environment variables (.env):

LMSTUDIO_API_BASE=http://localhost:1234
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=llama-2-7b-chat


Start the app:

streamlit run app/ui/app.py

Docker

Build the image:

docker build -t project-report-analyzer .


Run the container:

docker run -p 8501:8501 --env-file .env project-report-analyzer

Versioning

The project follows Semantic Versioning:

v0.1.0 – Core RAG system + case study

v0.2.0 – Chat interface and streaming-ready pipeline

Tags are pushed directly using standard Git workflows.

Limitations & Future Improvements
Current Limitations

Local FAISS index (single-node)

No reranking or confidence scoring

LLM quality depends on locally hosted model

Planned Enhancements

Token-level streaming in UI

Advanced reranking strategies

Cloud vector database

S3-based document persistence

LangSmith-based observability

Conclusion

This project demonstrates an end-to-end, production-oriented implementation of a RAG-based document analysis system, handling real-world complexities such as multi-structure PDFs, multi-document retrieval, and explainable AI outputs. The design emphasizes correctness, transparency, and extensibility, aligning with enterprise and assessment expectations.
