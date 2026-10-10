import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rtrival.rtrival import search
import os

load_dotenv()


llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=150,
            temperature=0
        )


PROMPT = """Answer the question using only the context below.
If the answer is not in the context, say you don't know.
Give short and crip answer

Context:
{context}

Question: {question}

Answer:"""


def build_context(results):
    return "\n\n".join(
        f"[{i + 1}] ({hit.get('source') or 'unknown'}) {hit['text']}"
        for i, hit in enumerate(results)
    )


def answer(question, results):
    context = build_context(results)

    response = llm.invoke(PROMPT.format(context=context, question=question))

    return {
        "answer": response.content,
        "sources": [hit.get("source") for hit in results],
        "matches": results,
    }


def ask(question, top_k=5):
    results = search(question, top_k=top_k)
    return answer(question, results)


if __name__ == "__main__":
    result = ask("What is this document about?")

    print(result["answer"])
    print("Sources:", result["sources"])
