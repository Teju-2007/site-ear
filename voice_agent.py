import os
from pathlib import Path

import requests
import sounddevice as sd
from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    BeginEvent,
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    TerminationEvent,
    TurnEvent,
)

from ml_pipeline import predict_clip


load_dotenv()

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

SITE_AUDIO_FILE = Path(
    os.getenv("SITE_AUDIO_FILE", "site_audio_clip.wav")
)

MODEL_FILE = os.getenv(
    "MODEL_FILE",
    "acoustic_model.joblib",
)

ANOMALY_THRESHOLD = float(
    os.getenv("ANOMALY_THRESHOLD", "0.70")
)

NORMAL_PHRASES = (
    "sounds fine",
    "sounds normal",
    "all clear",
    "no issue",
)

ANOMALY_PHRASES = (
    "hissing",
    "leak",
    "strange noise",
    "abnormal",
    "weird sound",
)

MANUAL_FLAG_PHRASES = (
    "flag this manually",
    "site ear flag this",
    "log this as an issue",
)


def post_to_backend(path: str, payload: dict):
    response = requests.post(
        f"{BACKEND_URL}{path}",
        json=payload,
        timeout=5,
    )
    response.raise_for_status()


def raise_flag(reason: str, confidence: float):
    post_to_backend(
        "/events/flag",
        {
            "reason": reason,
            "confidence": confidence,
        },
    )


def log_entry(
    text: str,
    label: str,
    confidence: float,
    agreement: bool,
):
    post_to_backend(
        "/events/log",
        {
            "text": text,
            "label": label,
            "confidence": confidence,
            "agreement": agreement,
        },
    )


def ask_followup(question: str):
    post_to_backend(
        "/events/followup",
        {
            "question": question,
        },
    )


def worker_claims_normal(text: str):
    text = text.lower()

    if any(phrase in text for phrase in ANOMALY_PHRASES):
        return False

    if any(phrase in text for phrase in NORMAL_PHRASES):
        return True

    return None


def check_manual_override(text: str) -> bool:
    text = text.lower()
    return any(
        phrase in text
        for phrase in MANUAL_FLAG_PHRASES
    )


def handle_transcript(text: str):
    print(f"Worker said: {text}")

    if check_manual_override(text):
        raise_flag(
            reason=f'Manually flagged by worker: "{text}"',
            confidence=1.0,
        )
        return

    if not SITE_AUDIO_FILE.exists():
        print(
            f"Missing site audio file: {SITE_AUDIO_FILE}"
        )
        return

    if not Path(MODEL_FILE).exists():
        print(
            f"Missing model file: {MODEL_FILE}. "
            "Run: python ml_pipeline.py train"
        )
        return

    try:
        label, confidence = predict_clip(SITE_AUDIO_FILE, MODEL_FILE)
    except Exception as error:
        print(f"Unable to analyse site audio: {error}")
        return

    print(
        f"Acoustic result: {label} "
        f"(confidence: {confidence:.2f})"
    )

    model_says_anomaly = (
        label == "anomalous"
        and confidence >= ANOMALY_THRESHOLD
    )

    worker_says_normal = worker_claims_normal(text)

    if worker_says_normal is True and model_says_anomaly:
        raise_flag(
            reason=(
                "Worker reported normal, but the acoustic "
                "model detected an anomaly "
                f"(confidence {confidence:.2f})."
            ),
            confidence=confidence,
        )

        ask_followup(
            "I am picking up an unusual acoustic signature "
            "here. Can you check that area again?"
        )

    elif worker_says_normal is False and not model_says_anomaly:
        log_entry(
            text=text,
            label=label,
            confidence=confidence,
            agreement=False,
        )

    else:
        log_entry(
            text=text,
            label=label,
            confidence=confidence,
            agreement=True,
        )


def on_begin(client, event: BeginEvent):
    print(f"AssemblyAI session started: {event.id}")


def on_turn(client, event: TurnEvent):
    if not event.transcript:
        return

    # Show the partial transcript in the terminal.
    print(event.transcript, end="\r")

    # Only process finalized turns.
    if event.end_of_turn:
        print()
        handle_transcript(event.transcript)


def on_terminated(client, event: TerminationEvent):
    print(
        "\nAssemblyAI session ended. "
        f"Processed {event.audio_duration_seconds} seconds."
    )


def on_error(client, error: RealTimeError):
    print(f"AssemblyAI error: {error}")


def main():
    api_key = os.getenv("ASSEMBLYAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is missing from the root .env file."
        )

    if not Path(MODEL_FILE).exists():
        raise RuntimeError(
            "acoustic_model.joblib is missing. "
            "Run python ml_pipeline.py train first."
        )

    client = RealTimeTranscriber(
        RealTimeTranscriberOptions(),
        api_key=api_key,
    )

    client.on(RealTimeEvents.Begin, on_begin)
    client.on(RealTimeEvents.Turn, on_turn)
    client.on(RealTimeEvents.Termination, on_terminated)
    client.on(RealTimeEvents.Error, on_error)

    client.connect(
        RealTimeParameters(
            speech_model="universal-3-5-pro",
            sample_rate=16_000,
        )
    )

    print("Listening. Press Ctrl+C to stop.")

    try:
        with sd.RawInputStream(
            samplerate=16_000,
            blocksize=800,
            channels=1,
            dtype="int16",
        ) as microphone:
            while True:
                audio_chunk, _ = microphone.read(800)
                client.stream(bytes(audio_chunk))

    except KeyboardInterrupt:
        print("\nStopping voice agent...")

    finally:
        client.disconnect(terminate=True)


if __name__ == "__main__":
    main()
