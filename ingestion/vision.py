import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from ingestion.indx import create_or_return_index, embedding_model


llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

CAPTION_PROMPT = (
    "Describe this image in detail for a text search index. "
    "If it is a chart or table, transcribe the key values, labels and axes."
)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_image(image_path):
    image_data = encode_image(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": CAPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


def index_images(image_paths, id_prefix, source=None):
    vectors = []

    for i, image_path in enumerate(image_paths, start=1):
        caption = caption_image(image_path)
        embedding = list(embedding_model.embed([caption]))[0]

        image_id = f"{id_prefix}-img-{i}"
        vectors.append(
            {
                "id": image_id,
                "values": embedding.tolist(),
                "metadata": {
                    "id": image_id,
                    "source": source,
                    "type": "image",
                    "image": Path(image_path).name,
                    "text": caption,
                },
            }
        )

    index = create_or_return_index()
    index.upsert(vectors=vectors)

    return len(vectors)
