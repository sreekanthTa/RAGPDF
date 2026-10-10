import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtrival.rtrival import search
from aug.aug import answer
import ragas_compat  # noqa: F401  patches langchain_community for ragas
from ragas import SingleTurnSample
from ragas import EvaluationDataset
from ragas.metrics import ( 
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
    )
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class ModelNamedEmbeddings:
    """Report a string ``model`` for ragas's usage event.

    ragas reads ``getattr(embeddings, "model", None)``, but FastEmbedEmbeddings
    exposes the fastembed model *object* there, which fails ragas's validation and
    makes answer_relevancy NaN. This delegates the calls and reports a string name.
    """

    def __init__(self, embeddings, model_name):
        self._embeddings = embeddings
        self.model = model_name

    def embed_query(self, text):
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts):
        return self._embeddings.embed_documents(texts)


def run_rag(queries, top_k=5):
    if isinstance(queries, str):
        queries = [queries]

    results = []

    for query in queries:
        hits = search(query, top_k=top_k)
        answer_text = answer(query, hits)["answer"]

        results.append(
            {
                "question": query,
                "contexts": [hit["text"] for hit in hits],
                "answer": answer_text,
            }
        )

    return results


if __name__ == "__main__":
    from reference_questions import REFERENCE_QUESTIONS

    evaluation_results = []

    for item in REFERENCE_QUESTIONS:
        result = run_rag(item["question"])[0]

        evaluation_results.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth"],
            }
        )

    print(f"Completed: {len(evaluation_results)} questions")

    samples = []

    for result in evaluation_results:
        sample = SingleTurnSample(
        user_input=result["question"],
        retrieved_contexts=result["contexts"],
        response=result["answer"],
        reference=result["ground_truth"]
         )

        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    # print("Samp", dataset[0])

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024,
            reasoning_effort="low",
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        ModelNamedEmbeddings(
            FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
            "BAAI/bge-small-en-v1.5",
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print(result)
    print(result.to_pandas())

