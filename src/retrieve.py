"""Load the local FAISS index and retrieve the most similar chunks."""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Tiny LangChain adapter around a local Sentence Transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def load_vector_store() -> FAISS:
    """Load the FAISS index created by ingest.py."""
    index_file = VECTOR_STORE_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python ingest.py` before starting the app."
        )

    embeddings = LocalSentenceTransformerEmbeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_documents(query: str, vector_store: FAISS, k: int = TOP_K):
    """Return the top-k chunks for a question."""
    return vector_store.similarity_search(query, k=k)


def format_source(document) -> str:
    """Format a document's source metadata for display."""
    metadata = document.metadata
    source = metadata.get("source", "unknown source")
    if "page_number" in metadata:
        return f"{source}, page {metadata['page_number']}"
    if "row" in metadata:
        return f"{source}, row {metadata['row']}"
    return source


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.retrieve \"your question\"")

    store = load_vector_store()
    for number, document in enumerate(
        retrieve_documents(" ".join(sys.argv[1:]), store), start=1
    ):
        print(f"{number}. {format_source(document)}")
        print(document.page_content[:300].replace("\n", " "))
