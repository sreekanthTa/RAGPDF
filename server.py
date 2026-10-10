import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ingestion.parser import extract_doc, save_results

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAGPDF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def _parse_pdf(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    job_id = uuid.uuid4().hex[:8]
    stem = Path(file.filename).stem
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = extract_doc(str(upload_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    output_name = f"{job_id}_{stem}"
    output_dir = OUTPUT_DIR / output_name
    save_results(result, str(output_dir))

    return result, output_name, output_dir, upload_path


def _cleanup(upload_path: Path, output_dir: Path) -> None:
    upload_path.unlink(missing_ok=True)
    shutil.rmtree(output_dir, ignore_errors=True)


def _extract_payload(filename: str, result: dict, output_name: str) -> dict:
    return {
        "filename": filename,
        "output_dir": output_name,
        "markdown": result["markdown"],
        "txt": result["txt"],
        "tables": [
            f"/outputs/{output_name}/tables/table_{i}.csv"
            for i in range(1, len(result["tables"]) + 1)
        ],
        "pictures": [
            f"/outputs/{output_name}/pictures/picture_{i}.png"
            for i in range(1, len(result["pictures"]) + 1)
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(file: UploadFile = File(...)):
    result, output_name, _, _ = _parse_pdf(file)
    return _extract_payload(file.filename, result, output_name)


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    from ingestion.indx import index_markdown

    result, output_name, output_dir, upload_path = _parse_pdf(file)

    try:
        stats = index_markdown(
            output_dir / "document.md",
            id_prefix=output_name,
            source=file.filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    image_count = 0
    picture_paths = sorted((output_dir / "pictures").glob("*.png"))
    if picture_paths:
        from ingestion.vision import index_images

        try:
            image_count = index_images(
                picture_paths,
                id_prefix=output_name,
                source=file.filename,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image indexing failed: {exc}") from exc

    _cleanup(upload_path, output_dir)

    return {
        "filename": file.filename,
        "markdown": result["markdown"],
        "txt": result["txt"],
        "tables": len(result["tables"]),
        "pictures": len(result["pictures"]),
        "indexed": stats,
        "images_indexed": image_count,
        "cleaned_up": True,
    }
