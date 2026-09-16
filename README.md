# Simple RAG Prototype

## Overview

Retrieval-Augmented Generation (RAG) first finds useful passages in a document collection, then gives those passages to a language model as context. The model answers from that context instead of relying only on what it learned during training.

## Architecture

```text
PDF / CSV
   ↓
Document Loader
   ↓
Chunking
   ↓
Local Embeddings
   ↓
FAISS
   ↓
Top-3 Retrieval
   ↓
Retrieved Context
   ↓
Groq LLM
   ↓
Answer + Sources
```

## Dataset

The repository contains three public research PDFs about RAG, sentence embeddings, and Transformers, plus the public Google Machine Learning Glossary CSV. See **DATA SOURCES.md** for the original URLs and usage information.

## Tech Stack

- Python 3.11+
- LangChain document loaders and text splitters
- Sentence Transformers: `sentence-transformers/all-MiniLM-L6-v2`
- FAISS for the local vector store
- Groq's official Python SDK
- `python-dotenv` for environment variables

## Setup

```bash
git clone <repo-url>
cd rag-prototype
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

The key is read from `GROQ_API_KEY`; it is never stored in the code.

## Build the Index

```bash
python ingest.py
```

The script finds PDFs and CSVs in `data/`, loads them into LangChain `Document` objects, splits them into chunks, creates local embeddings, and saves the FAISS index in `vector_store/` as `index.faiss` and `index.pkl`.

The current simple chunking settings are:

```text
chunk_size = 500
chunk_overlap = 50
```

The script prints the number of loaded documents, the number of chunks, and a few example chunks.

## Run the RAG Application

```bash
python app.py
```

Ask multiple questions. For each question the program embeds the question, retrieves the top 3 chunks, displays their sources, sends the retrieved context plus the question to Groq, and prints the answer.

Example questions:

- What is retrieval-augmented generation?
- What is a sentence embedding?
- What is self-attention?
- What is overfitting?

## Evaluate Retrieval

```bash
python evaluate.py
```

The evaluation file contains eight questions and an expected source filename. For each question, the script checks whether that source appears in the top 3 retrieved chunks and prints a simple source-hit `Precision@3` value.

## Architecture Explanation

1. **Document loading:** `PyPDFLoader` loads PDF pages. The CSV loader reads each glossary row as a searchable document. Source, page, and row metadata are preserved.
2. **Chunking:** `RecursiveCharacterTextSplitter` makes smaller overlapping text pieces.
3. **Embeddings:** Sentence Transformers converts every chunk and query into vectors locally.
4. **FAISS:** FAISS stores the chunk vectors for fast local similarity search.
5. **Retrieval:** The question is compared with the vectors and the top 3 chunks are returned.
6. **Context construction:** The retrieved chunks and their source labels are joined into one context string.
7. **Groq generation:** The official Groq SDK sends the context and question to `openai/gpt-oss-20b`, with instructions not to invent facts.

## Example

```text
Question:
What is a sentence embedding?

Retrieved Sources:
1. sentence_bert.pdf, page 1
2. sentence_bert.pdf, page 2
3. ml_glossary.csv, row 286

Answer:
The retrieved documents describe a sentence embedding as a vector representation
of a sentence that can be compared with other sentences for semantic similarity.
```

The exact pages and answer can vary slightly with the local index and model versions.

## Future Improvements

These are intentionally not implemented here:

- Better chunking
- Reranking
- Metadata filtering
- Pinecone or Weaviate
- Ragas
- MTEB
- Hybrid search

## References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks- Lewis et al. (2020)](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks- Reimers & Gurevych (2019)](https://arxiv.org/abs/1908.10084)
- [Attention Is All You Need- Vaswani et al. (2017)](https://research.google/pubs/attention-is-all-you-need/)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Documentation](https://python.langchain.com/)
