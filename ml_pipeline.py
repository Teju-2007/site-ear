from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
DATA_DIR = Path("data")
FEATURES_FILE = Path("features.csv")
MODEL_FILE = Path("acoustic_model.joblib")
LABELS = ("normal", "anomalous")
N_MFCC = 13
RANDOM_STATE = 42
def extract_mfcc(file_path: str | Path, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Turn one .wav file into one fixed-length MFCC vector."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    audio, sample_rate = librosa.load(str(file_path), sr=None, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {file_path}")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)
def build_dataset(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Read data/normal and data/anomalous and build features.csv data."""
    data_dir = Path(data_dir)
    rows = []
    for label in LABELS:
        class_dir = data_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {class_dir}. "
                f"Create data/normal and data/anomalous first."
            )
        audio_files = sorted(class_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found in {class_dir}")
        for audio_file in audio_files:
            try:
                features = extract_mfcc(audio_file)
                rows.append({
                    **{f"mfcc_{i}": float(value)
                       for i, value in enumerate(features)},
                    "label": label,
                    "file": str(audio_file),
                })
            except Exception as error:
                print(f"Skipping {audio_file}: {error}")
    if not rows:
        raise ValueError("No valid audio files were found.")
    return pd.DataFrame(rows)
def train_models(
    data_dir: str | Path = DATA_DIR,
    features_file: str | Path = FEATURES_FILE,
    model_file: str | Path = MODEL_FILE,
):
    df = build_dataset(data_dir)
    df.to_csv(features_file, index=False)
    print(f"Built dataset with {len(df)} audio samples")
    print(df["label"].value_counts().to_string())
    counts = df["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError(
            "Both classes need at least 2 clips. Add files to "
            "data/normal and data/anomalous."
        )
    feature_columns = [c for c in df.columns if c.startswith("mfcc_")]
    X = df[feature_columns]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = f1_score(
            y_test, predictions, pos_label="anomalous",
            average="binary", zero_division=0
        )
        results[name] = (model, score)
        print(f"\n{name} results:")
        print(classification_report(
            y_test, predictions, labels=list(LABELS), zero_division=0
        ))
        print(f"Anomalous F1 score: {score:.3f}")
    best_name = max(results, key=lambda name: results[name][1])
    best_model = results[best_name][0]
    joblib.dump(best_model, model_file)
    print(f"Selected model: {best_name}")
    print(f"Model saved to: {model_file}")
    return best_model
def predict_clip(
    file_path: str | Path,
    model_file: str | Path = MODEL_FILE,
    ) -> tuple[str, float]:
    """Return (normal/anomalous, confidence) for a new .wav file."""
    model_file = Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. Run the train command first."
        )
    model = joblib.load(model_file) 
    features = extract_mfcc(file_path).reshape(1, -1)
    prediction = str(model.predict(features)[0])
    confidence = float(np.max(model.predict_proba(features)[0]))
    return prediction, confidence
def main():
    parser = argparse.ArgumentParser(description="Site Ear acoustic pipeline")
    parser.add_argument("command", nargs="?", choices=("train", "predict"),
    default="train")
    parser.add_argument("audio_file", nargs="?")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-file", default=str(MODEL_FILE))
    args = parser.parse_args()
    if args.command == "train":
        train_models(args.data_dir, FEATURES_FILE, args.model_file)
    else:
        if not args.audio_file:
            parser.error("predict requires a .wav file")
        label, confidence = predict_clip(args.audio_file, args.model_file)
        print(f"Prediction: {label} (confidence: {confidence:.2f})")
if __name__ == "__main__":
    main()
