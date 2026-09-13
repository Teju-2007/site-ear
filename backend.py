from __future__ import annotations
from datetime import datetime, timezone
import csv, io, json, os, shutil, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = Path(os.getenv("SITE_EAR_LOG_FILE", BASE_DIR / "site_ear_log.jsonl"))
CLIPS_DIR = BASE_DIR / "flagged_clips"; CLIPS_DIR.mkdir(exist_ok=True)
SOURCE_AUDIO = BASE_DIR / os.getenv("SITE_AUDIO_FILE", "site-audio-clip.wav")
app = FastAPI(title="Site Ear API", version="1.1.0")
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
inspection_context = {"site_name": "Demo inspection", "location": "Site A", "worker_name": ""}

def now(): return datetime.now(timezone.utc).isoformat()
def load_events():
    if not LOG_FILE.exists(): return []
    result = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(event, dict): event.setdefault("id", uuid.uuid4().hex); result.append(event)
    return result
events = load_events()
def append_event(event):
    event.setdefault("id", uuid.uuid4().hex); event.setdefault("time", now()); event.setdefault("context", inspection_context.copy())
    events.append(event)
    with LOG_FILE.open("a", encoding="utf-8") as file: file.write(json.dumps(event) + "\n")
    return event
def severity_score(confidence): return round(1 + ((max(.5, min(float(confidence), 1)) - .5) / .5) * 9)
def severity_label(score): return "critical" if score >= 9 else "high" if score >= 7 else "medium" if score >= 4 else "low"
def save_clip(event_id):
    if not SOURCE_AUDIO.exists(): return None
    target = CLIPS_DIR / f"{event_id}.wav"; shutil.copy2(SOURCE_AUDIO, target); return f"/clips/{target.name}"
def raise_flag(reason, confidence=None, include_clip=False):
    event = {"id": uuid.uuid4().hex, "type": "flag", "reason": reason, "time": now()}
    if confidence is not None:
        event.update(confidence=confidence, severity=severity_score(confidence)); event["severity_label"] = severity_label(event["severity"])
    if include_clip: event["clip_url"] = save_clip(event["id"])
    return append_event(event)
def log_entry(text, label, confidence, agreement): return append_event({"type":"log", "transcript":text, "acoustic_label":label, "confidence":confidence, "agreement":agreement})
def ask_followup(question): return append_event({"type":"followup", "question":question})

class FlagIn(BaseModel): reason: str = Field(min_length=3, max_length=500); confidence: float | None = Field(default=None, ge=0, le=1); include_clip: bool = True
class LogIn(BaseModel): text: str; label: str; confidence: float = Field(ge=0, le=1); agreement: bool
class FollowupIn(BaseModel): question: str
class ContextIn(BaseModel): site_name: str = Field(default="Demo inspection", max_length=80); location: str = Field(default="Site A", max_length=80); worker_name: str = Field(default="", max_length=80)

@app.get("/health")
def health(): return {"status":"ok", "events_loaded":len(events)}
@app.get("/events")
def get_events(): return events
@app.get("/inspection/context")
def get_context(): return inspection_context
@app.put("/inspection/context")
def update_context(payload: ContextIn): inspection_context.update(payload.model_dump()); return inspection_context
@app.post("/events/flag")
def api_flag(payload: FlagIn): return raise_flag(payload.reason, payload.confidence, payload.include_clip)
@app.post("/events/log")
def api_log(payload: LogIn): return log_entry(payload.text, payload.label, payload.confidence, payload.agreement)
@app.post("/events/followup")
def api_followup(payload: FollowupIn): return ask_followup(payload.question)
@app.post("/demo/{scenario}")
def run_demo(scenario: str):
    if scenario == "mismatch":
        flag = raise_flag("Worker reported normal, but the acoustic model detected an anomaly.", .94, True)
        return {"scenario":scenario, "events":[flag, ask_followup("Unusual acoustic signature detected. Please inspect the valve area again.")]}
    if scenario == "concern": return {"scenario":scenario, "events":[log_entry("I can hear a hissing sound near the valve.", "normal", .78, False)]}
    raise HTTPException(404, "Unknown demo scenario")
@app.get("/events/summary")
def summary():
    flags = sum(e["type"] == "flag" for e in events); agreements = sum(e.get("agreement") is True for e in events)
    return {"total_events":len(events), "flags_raised":flags, "followups_asked":sum(e["type"] == "followup" for e in events), "agreements":agreements, "unresolved":max(0, flags-agreements), "summary_text":f"{len(events)} checks logged, {flags} flagged, {max(0, flags-agreements)} unresolved."}
@app.get("/events/confidence-trend")
def trend(): return [{"time":e["time"], "confidence":e["confidence"], "type":e["type"]} for e in events if "confidence" in e]
@app.get("/events/{event_id}/report.pdf")
def report(event_id: str):
    event = next((e for e in events if e.get("id") == event_id), None)
    if not event: raise HTTPException(404, "Inspection event not found")
    output = io.BytesIO(); document = SimpleDocTemplate(output, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54); styles = getSampleStyleSheet()
    ctx = event.get("context", {})
    rows = [["Inspection",ctx.get("site_name","")],["Location",ctx.get("location","")],["Worker",ctx.get("worker_name") or "Not recorded"],["Time (UTC)",event.get("time","")],["Severity",event.get("severity_label","review").upper()],["Confidence",f"{event.get('confidence',0)*100:.0f}%" if "confidence" in event else "Not scored"],["Finding",event.get("reason") or event.get("transcript") or event.get("question","")]]
    table = Table(rows, colWidths=[1.35*inch, 5.65*inch]); table.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#1b1d23")),("TEXTCOLOR",(0,0),(0,-1),colors.whitesmoke),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#b9bec8")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),8)]))
    document.build([Paragraph("Site Ear Incident Report", styles["Title"]),Spacer(1,12),Paragraph("Evidence-backed inspection summary",styles["Normal"]),Spacer(1,18),table]); output.seek(0)
    return StreamingResponse(output, media_type="application/pdf", headers={"Content-Disposition":f'attachment; filename="site-ear-incident-{event_id}.pdf"'})
@app.get("/events/export/json")
def export_json(): return StreamingResponse(io.StringIO(json.dumps(events,indent=2)), media_type="application/json", headers={"Content-Disposition":"attachment; filename=site_ear_log.json"})
@app.get("/events/export/csv")
def export_csv():
    buffer=io.StringIO(); fields=sorted({k for e in events for k in e if k != "context"}) or ["type","time"]; writer=csv.DictWriter(buffer,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(events)
    return StreamingResponse(iter([buffer.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=site_ear_log.csv"})
