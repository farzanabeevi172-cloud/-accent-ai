import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset


ACCENT_LABELS = {
    "en-IN": "Indian",
    "en-US": "American",
    "en-GB": "British",
}


def _extract_audio_array(example: Dict) -> Tuple[np.ndarray, int]:
    """
    Extract a mono audio array and sampling rate from a Common Voice example.

    Handles multiple possible column layouts to be robust across dataset
    versions (audio/voice/path).
    """
    if "audio" in example and example["audio"] is not None:
        audio = example["audio"]
        # Hugging Face common_voice typically provides dict with "array" and "sampling_rate"
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])

    if "voice" in example and example["voice"] is not None:
        audio = example["voice"]
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])

    # Fallback to loading from a file path if provided
    for key in ("path", "file", "filename"):
        if key in example and example[key]:
            wav, sr = librosa.load(example[key], sr=None, mono=True)
            return wav, sr

    raise ValueError("Could not extract audio from example; no known audio fields found.")


def _infer_locale(example: Dict) -> str:
    """
    Infer the locale code (en-IN, en-US, en-GB) from available metadata.

    Common Voice uses a language config (e.g. 'en') plus optional 'accent' field.
    This helper maps common accent metadata into the desired 3-way locale codes.
    """
    # If dataset already has a locale code, prefer it
    if "locale" in example and example["locale"]:
        return example["locale"]

    accent = (example.get("accent") or "").strip().lower()

    if any(token in accent for token in ["indian", "india", "hindi", "tamil", "telugu"]):
        return "en-IN"
    if any(token in accent for token in ["us", "american", "usa", "united states"]):
        return "en-US"
    if any(token in accent for token in ["england", "british", "uk", "united kingdom", "scotland", "wales"]):
        return "en-GB"

    # Unknown or unsupported accent
    return ""


def prepare_samples(
    dataset_name: str,
    dataset_config: str,
    output_dir: str,
    min_duration: float = 3.0,
    target_sr: int = 16000,
    max_per_class: int = 2000,
    seed: int = 42,
) -> None:
    """
    Build a balanced, preprocessed dataset from Mozilla Common Voice.

    - Loads Common Voice English subset from Hugging Face.
    - Keeps only en-IN, en-US, en-GB accents.
    - Enforces minimum duration (seconds) after resampling.
    - Balances classes by downsampling to the smallest class.
    - Saves 16kHz mono WAVs and train/val CSV manifests.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_root = Path(output_dir)
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {dataset_name} ({dataset_config}) from Hugging Face...")
    ds = load_dataset(dataset_name, dataset_config, split="train+validation")

    print("Filtering to supported accents (en-IN, en-US, en-GB)...")

    def accent_filter(example):
        locale = _infer_locale(example)
        return locale in ACCENT_LABELS

    ds = ds.filter(accent_filter)
    print(f"Filtered dataset size: {len(ds)}")

    if len(ds) == 0:
        raise RuntimeError(
            "No samples found for the requested accents. "
            "Please verify that you have access to Common Voice and that "
            "the dataset version contains en-IN, en-US, and en-GB examples."
        )

    # Group examples by locale
    grouped: Dict[str, List[int]] = {"en-IN": [], "en-US": [], "en-GB": []}
    for idx, ex in enumerate(ds):
        loc = _infer_locale(ex)
        if loc in grouped:
            grouped[loc].append(idx)

    for loc, indices in grouped.items():
        print(f"{loc}: {len(indices)} raw samples before duration filtering")

    def duration_ok(example) -> bool:
        try:
            wav, sr = _extract_audio_array(example)
        except Exception:
            return False
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        dur = float(len(wav)) / float(target_sr)
        return dur >= min_duration

    print(f"Applying duration filter (min_duration={min_duration}s)...")
    ds = ds.filter(duration_ok)
    print(f"Dataset size after duration filter: {len(ds)}")

    # Rebuild grouped indices after filtering
    grouped = {"en-IN": [], "en-US": [], "en-GB": []}
    for idx, ex in enumerate(ds):
        loc = _infer_locale(ex)
        if loc in grouped:
            grouped[loc].append(idx)

    for loc, indices in grouped.items():
        print(f"{loc}: {len(indices)} duration-validated samples")

    # Determine balanced size per class
    class_sizes = {loc: len(idxs) for loc, idxs in grouped.items()}
    min_class_size = min(size for size in class_sizes.values() if size > 0)
    per_class = min(min_class_size, max_per_class)
    print(f"Using {per_class} samples per class (balanced).")

    # Sample indices per class
    balanced_indices: List[Tuple[int, str]] = []
    for loc, idxs in grouped.items():
        if len(idxs) == 0:
            raise RuntimeError(f"No samples remaining for {loc} after filtering.")
        chosen = random.sample(idxs, k=per_class)
        for idx in chosen:
            balanced_indices.append((idx, loc))

    random.shuffle(balanced_indices)

    # Train/validation split (80/20)
    train_indices: List[Tuple[int, str]] = []
    val_indices: List[Tuple[int, str]] = []
    per_class_train_target: Dict[str, int] = {loc: int(0.8 * per_class) for loc in grouped}
    per_class_train_count: Dict[str, int] = {loc: 0 for loc in grouped}

    for idx, loc in balanced_indices:
        if per_class_train_count[loc] < per_class_train_target[loc]:
            train_indices.append((idx, loc))
            per_class_train_count[loc] += 1
        else:
            val_indices.append((idx, loc))

    print(
        f"Final split sizes — "
        f"train: {len(train_indices)}, val: {len(val_indices)} "
        f"(per class target train: {per_class_train_target})"
    )

    # Save audio and manifests
    manifests = {
        "train": [],
        "val": [],
    }

    for split_name, indices in [("train", train_indices), ("val", val_indices)]:
        split_dir = audio_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, (ds_idx, loc) in enumerate(indices):
            ex = ds[ds_idx]
            wav, sr = _extract_audio_array(ex)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

            # Ensure mono float32
            if wav.ndim > 1:
                wav = librosa.to_mono(wav)
            wav = wav.astype(np.float32)

            label_name = ACCENT_LABELS[loc]
            label_id = list(ACCENT_LABELS.keys()).index(loc)

            fname = f"{split_name}_{loc.replace('-', '')}_{i:06d}.wav"
            fpath = split_dir / fname
            sf.write(str(fpath), wav, target_sr)

            manifests[split_name].append(
                {
                    "filepath": str(fpath),
                    "label": label_name,
                    "label_id": label_id,
                    "locale": loc,
                }
            )

    # Save CSV manifests
    import csv

    for split_name, rows in manifests.items():
        csv_path = output_root / f"metadata_{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label", "label_id", "locale"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved {split_name} manifest: {csv_path} ({len(rows)} samples)")

    # Also persist label mapping for training/inference
    import json

    label_map = {
        "id2label": {i: ACCENT_LABELS[k] for i, k in enumerate(ACCENT_LABELS.keys())},
        "label2id": {v: i for i, v in enumerate(ACCENT_LABELS.values())},
        "locales": list(ACCENT_LABELS.keys()),
    }
    with open(output_root / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label mapping to {output_root / 'label_mapping.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Common Voice English subset for 3-way accent detection (Indian, American, British)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_16_0",
        help="Hugging Face dataset name for Mozilla Common Voice.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="en",
        help="Dataset configuration/language code (e.g., 'en').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to store processed audio and manifests.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=3.0,
        help="Minimum duration (seconds) of samples to keep.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sampling rate for output WAV files.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=2000,
        help="Maximum number of samples per accent (after filtering).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_samples(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        min_duration=args.min_duration,
        target_sr=args.sample_rate,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

