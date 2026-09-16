"""Command-line RAG application."""

from src.generate import generate_answer
from src.retrieve import format_source, load_vector_store, retrieve_documents


def main() -> None:
    print("=" * 36)
    print("Simple RAG Prototype")
    print("=" * 36)

    try:
        vector_store = load_vector_store()
    except FileNotFoundError as error:
        print(error)
        return

    while True:
        question = input("\nEnter your question (or type 'exit'): ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        documents = retrieve_documents(question, vector_store, k=3)
        print("\nRetrieved Sources:")
        for number, document in enumerate(documents, start=1):
            print(f"{number}. {format_source(document)}")

        context = "\n\n".join(
            f"Source: {format_source(document)}\n{document.page_content}"
            for document in documents
        )
        try:
            answer = generate_answer(question, context)
        except RuntimeError as error:
            print(f"\nConfiguration error: {error}")
            continue

        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
