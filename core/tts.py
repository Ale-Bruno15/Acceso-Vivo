"""Síntesis de voz para la respuesta del banco + tiempos por palabra para sincronizar avatar y subtítulos.

Estrategia:
  1. gTTS (voz neural de Google, español latinoamericano) — requiere internet; resultado cacheado en disco.
  2. pyttsx3 (SAPI5 en Windows, offline) — respaldo si no hay red.

La sincronía fina NO se calcula aquí: el reproductor web usa `audio.currentTime` como reloj maestro y escala
las fracciones de tiempo (0..1) que devolvemos a la duración real del audio. Así no hay deriva.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from config import TTS_CACHE_DIR


@dataclass
class Speech:
    audio: bytes
    mime: str        # "audio/mpeg" (gTTS) o "audio/wav" (pyttsx3)
    engine: str

    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.audio).decode('ascii')}"


def _cache_path(text: str, engine: str, ext: str) -> Path:
    key = hashlib.sha1(f"{engine}|{text}".encode("utf-8")).hexdigest()[:16]
    return TTS_CACHE_DIR / f"{engine}_{key}.{ext}"


def _gtts(text: str, lang: str, tld: str) -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang=lang, tld=tld, slow=False).write_to_fp(buf)
    return buf.getvalue()


def _pyttsx3(text: str) -> bytes:
    import pyttsx3

    engine = pyttsx3.init()
    for voice in engine.getProperty("voices"):  # preferir voz femenina en español (p. ej. "Sabina", "Helena")
        name = f"{voice.name} {voice.id}".lower()
        if "spanish" in name or "es-" in name or "español" in name:
            engine.setProperty("voice", voice.id)
            if any(n in name for n in ("sabina", "helena", "laura", "female")):
                break
    engine.setProperty("rate", 165)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        engine.stop()
        return out.read_bytes()


def synthesize(text: str, lang: str = "es", tld: str = "com.mx", prefer: str = "gtts") -> Speech:
    """Devuelve el audio de `text`, desde caché si ya existe."""
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    order = ["gtts", "pyttsx3"] if prefer == "gtts" else ["pyttsx3", "gtts"]
    errors = []
    for engine in order:
        ext, mime = ("mp3", "audio/mpeg") if engine == "gtts" else ("wav", "audio/wav")
        path = _cache_path(f"{lang}|{tld}|{text}", engine, ext)
        if path.exists():
            return Speech(path.read_bytes(), mime, engine)
        try:
            audio = _gtts(text, lang, tld) if engine == "gtts" else _pyttsx3(text)
            path.write_bytes(audio)
            return Speech(audio, mime, engine)
        except Exception as exc:  # sin red, sin voz SAPI, etc.
            errors.append(f"{engine}: {exc}")
    raise RuntimeError("No se pudo sintetizar la voz -> " + " | ".join(errors))


# --- Tiempos estimados por palabra --------------------------------------------
_VOWEL_GROUPS = re.compile(r"[aeiouáéíóúü]+", re.IGNORECASE)


def _syllables(word: str) -> int:
    return max(1, len(_VOWEL_GROUPS.findall(word)))


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def word_timings(text: str, pause_weight: float = 1.5) -> list[dict]:
    """Reparte la frase en fracciones [start, end] (0..1) proporcionales a las sílabas.

    Las comas y puntos suman un peso de pausa. El navegador multiplica por `audio.duration`.
    """
    tokens = re.findall(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+|[.,;:!?¿¡]", text)
    weights = [(_syllables(t) if t[0].isalnum() else pause_weight * (t in ".,;:!?")) for t in tokens]
    total = sum(weights) or 1.0
    out, acc = [], 0.0
    for tok, w in zip(tokens, weights):
        start = acc / total
        acc += w
        if tok[0].isalnum():
            out.append({"word": tok, "start": round(start, 4), "end": round(acc / total, 4)})
    return out
