
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import os


load_dotenv()


embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


pinecone = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


INDEX_NAME = "PDF_Index"


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
            wait=True,
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


def index_markdown(markdown_path, chunk_size=400, overlap=50, id_prefix=None, source=None):
    chunks = create_chunks(
        markdown_path,
        chunk_size=chunk_size,
        overlap=overlap,
        id_prefix=id_prefix,
        source=source,
    )
    embedded_chunks = create_embeddings(chunks)
    index = create_or_return_index()
    upserted = upsert_to_pinecone(embedded_chunks, index)

    return {
        "index": INDEX_NAME,
        "chunks": upserted,
        "id_prefix": id_prefix or Path(markdown_path).stem,
    }


def main():
    currn_path = Path(__file__).resolve().parent
    currnt_file = currn_path / "rn_output" / "document.md"

    stats = index_markdown(currnt_file)

    print(stats)


if __name__ == "__main__":
    main()
