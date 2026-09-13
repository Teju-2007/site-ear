# Site Ear

Site Ear cross-checks a worker's spoken inspection statement against a local
acoustic anomaly model, and surfaces any mismatch — worker says "sounds
fine" but the model hears an anomaly, or vice versa — as a flagged event on
a live dashboard.

**🔗 Live demo:** [site-ear.vercel.app](https://site-ear.vercel.app)
**🔗 API:** [site-ear-api.onrender.com](https://site-ear-api.onrender.com)

> Note: the API runs on Render's free tier and may take 30–60 seconds to
> wake up if it's been idle. If the demo looks slow to load at first, give
> it a moment — it's just spinning up.

---

## How it works

1. A local voice agent (`voice_agent.py`) transcribes a worker's spoken
   inspection notes in real time via AssemblyAI.
2. In parallel, a scikit-learn model (`ml_pipeline.py` / `acoustic_model.joblib`)
   scores a site audio clip for acoustic anomalies.
3. If the worker's words and the model's verdict disagree, the FastAPI
   backend logs a flagged event — with confidence score, audio clip, and an
   auto-generated incident PDF.
4. The React dashboard shows a live timeline, confidence trend chart, and
   exportable evidence (CSV/JSON/PDF) for every inspection.

## Architecture

```
voice_agent.py  →  FastAPI backend  →  React dashboard
(mic + AssemblyAI   (events, PDF        (site-ear-dashboard,
 + local ML model)   reports, CORS)      Vite + React)
```

## Run locally

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```
3. Train the model:
   ```
   python ml_pipeline.py train
   ```
4. Start the backend (in one terminal):
   ```
   uvicorn backend:app --reload --port 8000
   ```
5. Start the dashboard (in a second terminal):
   ```
   cd site-ear-dashboard
   npm install
   npm run dev
   ```
6. Start the voice agent (with a valid `ASSEMBLYAI_API_KEY` in `.env`):
   ```
   python voice_agent.py
   ```

## Deploy

**Backend (Render):**
- Deploy using the included `render.yaml`.
- It installs dependencies and trains the committed audio dataset during
  the build, then starts FastAPI on Render's assigned port.
- Set `FRONTEND_ORIGINS` to the final Vercel URL — **no trailing slash**
  (e.g. `https://site-ear.vercel.app`, not `.../`), since CORS origin
  matching is an exact string match.

**Frontend (Vercel):**
- Set the project root to `site-ear-dashboard`.
- Set `VITE_API_URL` to the public Render API URL before deploying.

**Security note:** Never set `ASSEMBLYAI_API_KEY` on Vercel — it should only
ever live in the `.env` on the machine running `voice_agent.py`, since that's
the only place doing live microphone transcription.

## Tech stack

- **Voice/ML:** AssemblyAI streaming transcription, scikit-learn, librosa
- **Backend:** FastAPI, ReportLab (PDF generation)
- **Frontend:** React + Vite
- **Hosting:** Render (API) + Vercel (dashboard)
