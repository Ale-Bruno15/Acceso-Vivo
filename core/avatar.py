"""Puente Python -> avatar 3D (Three.js dentro de un componente HTML de Streamlit).

Python decide QUÉ decir (texto, glosas LSRD, audio); el navegador decide CÓMO moverse (IK de brazos,
configuraciones manuales, blendshapes faciales) usando el audio como reloj maestro.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import config
from core.tts import Speech, strip_accents, word_timings

_TEMPLATE = Path(__file__).with_name("avatar_player.html")


@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    return json.loads(config.LEXICON_PATH.read_text(encoding="utf-8"))


def text_to_gloss(text: str) -> list[str]:
    """Español -> glosas por diccionario (MVP). Palabras sin seña se omiten; repetidas consecutivas se colapsan.

    Producción: reemplazar por un traductor español->glosa LSRD validado por intérpretes.
    """
    glosses: list[str] = []
    for word in re.findall(r"[\wñ]+", strip_accents(text.lower())):
        g = config.WORD_TO_GLOSS.get(word)
        if g and (not glosses or glosses[-1] != g):
            glosses.append(g)
    return glosses


def build_script(text: str, glosses: list[str], speech: Speech | None) -> dict:
    """Guion que consume el reproductor: glosas válidas, tiempos por palabra y audio embebido."""
    known = load_lexicon()["glosses"]
    return {
        "text": text,
        "glosses": [g for g in glosses if g in known],
        "missing": [g for g in glosses if g not in known],
        "words": word_timings(text),
        "audio": speech.data_uri() if speech else None,
        "engine": speech.engine if speech else None,
        "id": time.time_ns(),  # fuerza reinicio aunque el texto se repita
    }


def render_avatar(script: dict | None = None, height: int = 640) -> None:
    """Renderiza el avatar. Con `script=None` queda en reposo (respiración, parpadeo)."""
    payload = {
        "lexicon": load_lexicon(),
        "script": script,
        "glbUrl": config.AVATAR_GLB_URL if config.AVATAR_GLB.exists() else None,
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = _TEMPLATE.read_text(encoding="utf-8").replace("__PAYLOAD__", data)
    if hasattr(st, "iframe"):   # Streamlit >= 1.5x: reemplazo oficial de components.html
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)
