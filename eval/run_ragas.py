import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import ragas_compat  # noqa: F401,E402  patches langchain_community for ragas
from langchain_community.embeddings import FastEmbedEmbeddings  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def load_dataset(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))

    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=row["ground_truth"],
            )
            for row in rows
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(HERE / "dataset.json"))
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    judge = LangchainLLMWrapper(ChatGroq(model=args.model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )

    print(result)
    print(result.to_pandas())


if __name__ == "__main__":
    main()
