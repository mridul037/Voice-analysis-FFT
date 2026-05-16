# Voice Analyser (scaffold)

Realtime **WebSocket** stream of **FFT** magnitudes from a **FastAPI** backend (NumPy), rendered in the browser with **WebGL2**.

## Layout

- `backend/` — FastAPI app (`/health`, `/ws` demo stream)
- `frontend/` — Vite + React + TypeScript + WebGL spectrum

## Run

**Terminal 1 — API**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — UI**

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The dev server proxies `/ws` and `/health` to the API on port 8000.

## Next steps

- Capture microphone in the browser and send PCM over WebSocket (binary frames).
- Run FFT on GPU (WebGPU / compute) or keep server-side workers for distributed processing.
- Add auth, room routing, and back-pressure for production loads.
