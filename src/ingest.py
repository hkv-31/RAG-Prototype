"""Load PDFs/CSV files, split them, embed them, and save a FAISS index."""

import csv
import html
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .retrieve import LocalSentenceTransformerEmbeddings, VECTOR_STORE_DIR


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_csv_text(value: str) -> str:
    """Make the public glossary's HTML definitions readable as plain text."""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pdf_documents(path: Path) -> list[Document]:
    documents = PyPDFLoader(str(path)).load()
    for document in documents:
        document.metadata["source"] = path.name
        if "page" in document.metadata:
            document.metadata["page_number"] = int(document.metadata["page"]) + 1
            document.metadata.pop("page", None)
    return documents


def load_csv_documents(path: Path) -> list[Document]:
    """Load the public Google glossary (pipe-delimited term/definition rows)."""
    documents: list[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="|")
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue

            term = row[0].strip()
            definition = clean_csv_text(row[1])
            if not term or not definition:
                continue
            if term.lower().startswith("introduction"):
                continue

            term = term.replace(" (Google Machine Learning Glossary)", "")
            documents.append(
                Document(
                    page_content=f"Term: {term}\nDefinition: {definition}",
                    metadata={"source": path.name, "row": row_number},
                )
            )
    return documents


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        documents.extend(load_pdf_documents(path))
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name == "evaluation_questions.csv":
            continue
        documents.extend(load_csv_documents(path))
    return documents


def build_index() -> None:
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print("\nExample chunks:")
    for number, chunk in enumerate(chunks[:3], start=1):
        preview = " ".join(chunk.page_content.split())
        preview = preview.encode("ascii", errors="replace").decode("ascii")
        print(f"\nChunk {number} ({chunk.metadata}):\n{preview[:400]}")

    print("\nCreating local embeddings...")
    embeddings = LocalSentenceTransformerEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"\nFAISS index saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_index()
