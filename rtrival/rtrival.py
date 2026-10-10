import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.indx import create_or_return_index, embedding_model, search_bm25


def embed_query(query: str) -> list[float]:
    return list(embedding_model.embed([query]))[0].tolist()


def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            scores[hit["id"]] = scores.get(hit["id"], 0) + 1.0 / (k+ rank + 1)

    return sorted(scores.items(), key = lambda item: item[1], reverse = True)



def search(query: str, top_k: int = 2):
    index = create_or_return_index()

    # Dense (embedding) search in Pinecone
    emb_res = index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_metadata=True,
    )

    dense = [
        {
            "id": match.id,
            "score": match.score,
            "text": (match.metadata or {}).get("text"),
            "source": (match.metadata or {}).get("source"),
        }
        for match in emb_res.matches
    ]

    # BM25 lexical search over the persisted store (empty until an ingest runs)
    bm_res = search_bm25(query, top_k=top_k)

    # Fuse dense + BM25 rankings
    by_id = {hit["id"]: hit for hit in bm_res}
    by_id.update({hit["id"]: hit for hit in dense})

    fused = [
        {**by_id[hit_id], "rrf_score": score}
        for hit_id, score in reciprocal_rank_fusion([dense, bm_res])
        if hit_id in by_id
    ]

    # TODO: rerank the fused candidates here (cross-encoder / LLM reranker)
    # before trimming to top_k.
    return fused[:top_k]


if __name__ == "__main__":
    hits = search("What is this document about?")

    for hit in hits:
        print(f"[{hit['score']:.4f}] {hit['id']} ({hit['source']})")
        print(hit["text"])
        print("-" * 40)
