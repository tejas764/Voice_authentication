import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import parameters as p
from feature_extraction import get_embedding
from model_utils import load_voice_model


class VoiceAuthService:
    def __init__(self, model_file: str = p.MODEL_FILE, embed_dir: str = p.EMBED_LIST_FILE):
        self.model_file = model_file
        self.embed_dir = Path(embed_dir)
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.embed_dir / "users.csv"
        self._metadata_cache: Optional[pd.DataFrame] = None
        self._metadata_mtime: Optional[float] = None
        self._embed_cache_names: List[str] = []
        self._embed_cache_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._embed_cache_mtimes: Dict[str, float] = {}
        self._embed_cache_valid = False
        self.model = load_voice_model(model_file)
        self._ensure_metadata()

    def _ensure_metadata(self) -> None:
        if not self.meta_file.exists():
            df = pd.DataFrame(columns=["speaker_key", "name", "register_number", "file", "created_at"])
            df.to_csv(self.meta_file, index=False)

    def _load_metadata(self) -> pd.DataFrame:
        self._ensure_metadata()
        mtime = self.meta_file.stat().st_mtime
        if self._metadata_cache is not None and self._metadata_mtime == mtime:
            return self._metadata_cache.copy()
        self._metadata_cache = pd.read_csv(self.meta_file)
        self._metadata_mtime = mtime
        return self._metadata_cache.copy()

    def _save_metadata(self, df: pd.DataFrame) -> None:
        df.to_csv(self.meta_file, index=False)
        self._metadata_cache = df.copy()
        self._metadata_mtime = self.meta_file.stat().st_mtime

    def _invalidate_embed_cache(self) -> None:
        self._embed_cache_valid = False

    def _refresh_embed_cache_if_needed(self) -> Tuple[List[str], np.ndarray]:
        files = sorted([f for f in self.embed_dir.glob("*.npy") if f.is_file()])
        current_mtimes = {f.stem: f.stat().st_mtime for f in files}

        if self._embed_cache_valid and current_mtimes == self._embed_cache_mtimes:
            return self._embed_cache_names, self._embed_cache_matrix

        names: List[str] = []
        vectors: List[np.ndarray] = []
        for f in files:
            vector = np.load(f).astype(np.float32, copy=False).reshape(-1)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            names.append(f.stem)
            vectors.append(vector)

        if vectors:
            matrix = np.vstack(vectors)
        else:
            matrix = np.empty((0, 0), dtype=np.float32)

        self._embed_cache_names = names
        self._embed_cache_matrix = matrix
        self._embed_cache_mtimes = current_mtimes
        self._embed_cache_valid = True
        return names, matrix

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "user"

    def _speaker_key(self, name: str, register_number: str) -> str:
        return f"{self._normalize(name)}__{self._normalize(register_number)}"

    def list_enrolled_users(self) -> List[str]:
        names, _ = self._refresh_embed_cache_if_needed()
        return names

    def list_enrolled_user_details(self) -> pd.DataFrame:
        df = self._load_metadata()
        if df.empty:
            return df
        return df.sort_values("created_at", ascending=False)

    def get_user_by_speaker_key(self, speaker_key: str) -> Optional[Dict[str, str]]:
        df = self._load_metadata()
        match = df[df["speaker_key"] == speaker_key]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "speaker_key": str(row["speaker_key"]),
            "name": str(row["name"]),
            "register_number": str(row["register_number"]),
            "file": str(row["file"]),
        }

    def enroll_person(self, name: str, register_number: str, audio_path: str) -> Dict[str, str]:
        if not register_number or not register_number.strip():
            raise ValueError("Register number is required for enrollment.")
        speaker_key = self._speaker_key(name, register_number)
        out_path = self.enroll(speaker_key, audio_path)

        df = self._load_metadata()
        df = df[df["speaker_key"] != speaker_key]
        df.loc[len(df)] = {
            "speaker_key": speaker_key,
            "name": name.strip(),
            "register_number": register_number.strip(),
            "file": out_path,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self._save_metadata(df)
        return {"speaker_key": speaker_key, "file": out_path}

    def enroll(self, name: str, audio_path: str) -> str:
        if not name or not name.strip():
            raise ValueError("Name is required for enrollment.")
        embedding = get_embedding(self.model, audio_path, p.MAX_SEC)
        out_path = self.embed_dir / f"{name.strip()}.npy"
        np.save(out_path, embedding)
        self._invalidate_embed_cache()
        return str(out_path)

    def enroll_csv(self, csv_path: str) -> int:
        df = pd.read_csv(csv_path)
        columns = [c.lower() for c in df.columns]

        if "filename" in columns and "speaker" in columns:
            filename_col = df.columns[columns.index("filename")]
            speaker_col = df.columns[columns.index("speaker")]
        elif len(df.columns) >= 2:
            filename_col = df.columns[0]
            speaker_col = df.columns[1]
        else:
            raise ValueError("CSV must contain at least two columns: filename and speaker.")

        count = 0
        for _, row in df.iterrows():
            audio_path = str(row[filename_col])
            speaker = str(row[speaker_col])
            self.enroll(speaker, audio_path)
            count += 1
        return count

    def recognize(self, audio_path: str) -> Dict[str, object]:
        speaker_names, enroll_matrix = self._refresh_embed_cache_if_needed()
        if not speaker_names:
            raise RuntimeError("No enrolled users found. Enroll at least one user first.")

        test_embedding = get_embedding(self.model, audio_path, p.MAX_SEC).astype(np.float32, copy=False).reshape(-1)
        test_norm = np.linalg.norm(test_embedding)
        if test_norm > 0:
            test_embedding = test_embedding / test_norm

        if p.COST_METRIC.lower() == "cosine":
            scores = 1.0 - np.dot(enroll_matrix, test_embedding)
        else:
            scores = np.linalg.norm(enroll_matrix - test_embedding, axis=1)

        distances = {name: float(score) for name, score in zip(speaker_names, scores)}

        speaker = min(distances, key=distances.get)
        score = distances[speaker]
        accepted = score < p.THRESHOLD

        return {
            "accepted": accepted,
            "speaker": speaker,
            "score": score,
            "threshold": p.THRESHOLD,
            "metric": p.COST_METRIC,
            "distances": distances,
            "matched_user": self.get_user_by_speaker_key(speaker),
        }
