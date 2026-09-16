"""Generate an answer from retrieved context using the Groq Python SDK."""

import os

from dotenv import load_dotenv
from groq import Groq


MODEL_NAME = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context.\n"
    "If the answer cannot be found in the provided context, say that the information "
    "is not available in the provided documents.\n"
    "Do not invent facts."
)


def generate_answer(question: str, retrieved_context: str) -> str:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    user_prompt = f"Context:\n{retrieved_context}\n\nQuestion:\n{question}\n\nAnswer:"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or "No answer was returned."
