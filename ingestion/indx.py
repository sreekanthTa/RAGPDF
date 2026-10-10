
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import os
import pickle
from rank_bm25 import BM25Okapi


load_dotenv()


embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


INDEX_NAME = "rag-pdf-index"
BM25_DIR = Path(__file__).resolve().parent.parent / "bm25_store"


def create_or_return_index():
    has_index = pinecone.has_index(INDEX_NAME)

    if not has_index:
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

    pc_index = pinecone.index(INDEX_NAME)

    return pc_index


def create_chunks(file_path, chunk_size=230, overlap=30, id_prefix=None, source=None):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # MarkDown Splitter
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    markdown_chunks_documents = markdown_splitter.split_text(text)

    # Recursive Splitter
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    txt_chunks_documents = txt_splitter.split_documents(markdown_chunks_documents)

    prefix = id_prefix or Path(file_path).stem
    for i, chunk in enumerate(txt_chunks_documents):
        chunk.metadata["id"] = f"{prefix}-{i}"
        if source is not None:
            chunk.metadata["source"] = source

    return txt_chunks_documents


def create_embeddings(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = list(embedding_model.embed(texts))

    for chunk, embedding in zip(chunks, embeddings):
        chunk.metadata["embedding"] = embedding

    return chunks


def upsert_to_pinecone(chunks, index):
    vectors = []

    for chunk in chunks:
        embedding = chunk.metadata.pop("embedding")

        vector = {
            "id": chunk.metadata["id"],
            "values": embedding.tolist(),
            "metadata": {
                **chunk.metadata,
                "text": chunk.page_content,
            }
        }

        vectors.append(vector)

    index.upsert(vectors=vectors)

    return len(vectors)


def create_bm25(chunks):
    tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def save_bm25(id_prefix, chunks):
    BM25_DIR.mkdir(exist_ok=True)
    with open(BM25_DIR / f"{id_prefix}.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(id_prefix=None):
    if not BM25_DIR.exists():
        return []

    if id_prefix is not None:
        files = [BM25_DIR / f"{id_prefix}.pkl"]
    else:
        files = sorted(BM25_DIR.glob("*.pkl"))

    chunks = []
    for file in files:
        if file.exists():
            with open(file, "rb") as f:
                chunks.extend(pickle.load(f))

    return chunks


def search_bm25(query, top_k=5, id_prefix=None):
    chunks = load_chunks(id_prefix)
    if not chunks:
        return []

    # NOTE: BM25 is rebuilt on every call. Cache it if this becomes a hot path.
    bm25 = create_bm25(chunks)
    scores = bm25.get_scores(query.lower().split())

    top_indices = scores.argsort()[-top_k:][::-1]

    return [
        {
            "id": chunks[index].metadata["id"],
            "text": chunks[index].page_content,
            "source": chunks[index].metadata.get("source"),
            "score": float(scores[index]),
        }
        for index in top_indices
    ]


def index_markdown(markdown_path, chunk_size=400, overlap=50, id_prefix=None, source=None):
    chunks = create_chunks(
        markdown_path,
        chunk_size=chunk_size,
        overlap=overlap,
        id_prefix=id_prefix,
        source=source,
    )
    save_bm25(id_prefix or Path(markdown_path).stem, chunks)
    print(
        "BM25 index created"
    )

    embedded_chunks = create_embeddings(chunks)
    print(
        "Embeddings created"
    )
    index = create_or_return_index()
    upserted = upsert_to_pinecone(embedded_chunks, index)

    return {
        "index": INDEX_NAME,
        "chunks": upserted,
        "id_prefix": id_prefix or Path(markdown_path).stem,
    }


def main():
    currn_path = Path(__file__).resolve().parent
    rn_output = currn_path / "rn_output"

    stats = index_markdown(rn_output / "document.md", source="rn.pdf")
    print(stats)

    picture_paths = sorted((rn_output / "pictures").glob("*.png"))
    if picture_paths:
        from ingestion.vision import index_images

        image_count = index_images(
            picture_paths,
            id_prefix=Path(rn_output / "document.md").stem,
            source="rn.pdf",
        )
        print("images indexed:", image_count)


if __name__ == "__main__":
    main()
