"""Simple top-3 source retrieval evaluation."""

import csv
from pathlib import Path

from src.retrieve import load_vector_store, retrieve_documents


ROOT_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = ROOT_DIR / "data" / "evaluation_questions.csv"


def main() -> None:
    vector_store = load_vector_store()
    with QUESTIONS_FILE.open("r", encoding="utf-8", newline="") as file:
        questions = list(csv.DictReader(file))

    correct = 0
    print("=" * 36)
    print("RAG Retrieval Evaluation")
    print("=" * 36)

    for item in questions:
        documents = retrieve_documents(item["question"], vector_store, k=3)
        sources = [Path(document.metadata.get("source", "")).name for document in documents]
        expected = item["expected_source"]
        hit = expected in sources
        correct += int(hit)
        status = "PASS" if hit else "MISS"
        print(f"[{status}] {item['question']} -> expected {expected}")
        print(f"      retrieved: {', '.join(sources)}")

    precision_at_3 = correct / len(questions) if questions else 0
    print(f"\nQuestions evaluated: {len(questions)}")
    print(f"Correct source retrieved: {correct}")
    print(f"Precision@3: {precision_at_3:.2f}")
    print("Note: this is a simple source-hit rate at top-3, not a full IR benchmark.")


if __name__ == "__main__":
    main()
