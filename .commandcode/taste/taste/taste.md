# Taste
- Prefers extraction/processing results persisted to disk rather than only returned in memory or printed to the console, each result type written to its own format-appropriate file (e.g. `.md`/`.txt`/`.json` for document output, `.csv` per table, `.png` per image). Confidence: 0.55
- Reports problems by pasting raw IDE context + the full stack trace with little to no surrounding explanation, expecting the agent to diagnose the root cause and apply the fix directly rather than asking them to describe the issue. Confidence: 0.6
- Keeps frontend and backend as separate projects — wants the React frontend in its own folder outside the backend repo rather than a single combined workspace. Confidence: 0.55
- Prefers exposing the processing pipeline through a thin HTTP API layer rather than importing pipeline code directly into the UI. Confidence: 0.5
- Prefers ReactJS (Vite) for frontend apps. Confidence: 0.55
- Prefers FastAPI (with uvicorn) over Flask for Python web APIs. Confidence: 0.6
- Wants asynchronous operations surfaced in the UI with a loading indicator followed by explicit success/failure states. Confidence: 0.5
- Concerned about intermediate artifacts accumulating on disk; prefers temporary inputs/outputs be cleaned up once their content has been persisted downstream (e.g. delete the uploaded PDF and parsed output after chunks are in the vector store) rather than leaving folders to grow unbounded. Confidence: 0.5

