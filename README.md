# RAGPDF (backend)

FastAPI service that extracts text, tables and images from uploaded PDFs using
[docling](https://github.com/docling-project/docling).

## Run

```powershell
.venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000
```

Endpoints:

- `GET /health` — health check
- `POST /extract` — multipart upload (`file`) of a PDF; returns markdown, txt,
  table CSV paths and picture PNG paths
- `GET /outputs/...` — serves the generated files

## Notes

- Extraction runs docling synchronously, so a request can take minutes on CPU
  (OCR). Set `pdf_options.do_ocr = False` in `ingestion/parser.py` for a big
  speed-up on PDFs that already have a text layer.
- The React frontend lives in a sibling folder: `../rag-frontend`.
