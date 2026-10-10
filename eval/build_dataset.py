import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from reference_questions import REFERENCE_QUESTIONS  # noqa: E402
from rtrival.rtrival import search  # noqa: E402
from aug.aug import answer  # noqa: E402


def build_dataset(top_k=5):
    rows = []

    for item in REFERENCE_QUESTIONS:
        hits = search(item["question"], top_k=top_k)

        rows.append(
            {
                "question": item["question"],
                "contexts": [hit["text"] for hit in hits],
                "answer": answer(item["question"], hits)["answer"],
                "ground_truth": item["ground_truth"],
            }
        )

    return rows


if __name__ == "__main__":
    rows = build_dataset()

    out = HERE / "dataset.json"
    out.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {len(rows)} rows to {out}")
