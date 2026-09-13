"""Script 3 — AccesoVivo: ventanilla bancaria inclusiva (Streamlit).

    streamlit run app.py

Entrada : cámara -> MediaPipe Holistic -> clasificador sklearn -> texto para el funcionario.
Salida  : respuesta del banco -> avatar 3D (Three.js) signando en LSRD + voz sincronizada (gTTS/pyttsx3).
"""
from __future__ import annotations

import sys
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
import streamlit as st

import config
from core.avatar import build_script, render_avatar, text_to_gloss
from core.tts import synthesize

st.set_page_config(page_title="AccesoVivo · Ventanilla inclusiva", page_icon="🤟",
                   layout="wide", initial_sidebar_state="collapsed")

# ───────────────────────── Estilos (réplica del diseño AccesoVivo) ─────────────────────────
st.markdown("""
<style>
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"] { display:none !important; }
.block-container { max-width: 1320px; padding-top: 1.6rem; padding-bottom: 1rem; }
html, body, [class*="css"] { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
iframe { border: 0 !important; border-radius: 24px; }

.av-brand { text-align:center; margin: 0 0 1.4rem; }
.av-logo { font-size: 30px; font-weight: 300; letter-spacing: .32em; color:#3A3A3A; }
.av-tag  { font-size: 11px; letter-spacing: .34em; color:#A09A99; margin-top: 2px; }
.av-section { font-size: 13px; font-weight: 600; letter-spacing: .3em; color:#3A3A3A; margin: 1.2rem 0 .9rem; }
.av-lead { color:#6F6866; font-size: 14px; line-height: 1.5; text-align:center; margin: 0 auto .8rem; max-width: 560px; }

/* Tarjetas de opción: el botón real queda invisible encima de la tarjeta */
div[class*="st-key-opt_"] { position: relative; }
.av-card { display:flex; align-items:center; gap:16px; padding: 20px 22px; border-radius: 16px;
  background: linear-gradient(100deg, #FFD3BC 0%, #FFC9C4 55%, #FFC2CA 100%);
  box-shadow: 0 10px 24px rgba(255,150,140,.25); transition: transform .15s, box-shadow .15s; position: relative; }
div[class*="st-key-opt_"]:hover .av-card { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(255,150,140,.34); }
.av-icon { flex: 0 0 46px; height: 46px; border-radius: 50%; background: rgba(255,255,255,.78);
  display:flex; align-items:center; justify-content:center; color:#4A3B39; }
.av-title { font-size: 15.5px; font-weight: 500; letter-spacing: .03em; color:#2F2A29; text-transform: uppercase; }
.av-desc { font-size: 12.5px; color:#6B5A57; margin-top: 3px; }
.av-badge { position:absolute; top:10px; right:12px; background:#3B3B3B; color:#fff; font-size:10px;
  font-weight:600; padding: 2px 8px; border-radius: 999px; }
div[class*="st-key-btn_"] { position: absolute !important; inset: 0; z-index: 3; margin: 0 !important;
  width: 100% !important; height: 100% !important; }
div[class*="st-key-btn_"] .stButton, div[class*="st-key-btn_"] .stButton > div,
div[class*="st-key-btn_"] button { width: 100% !important; height: 100% !important; }
div[class*="st-key-btn_"] button { opacity: 0; cursor: pointer; }

/* Cámara */
.av-cam { background:#1A1A1A; border-radius: 18px; aspect-ratio: 4/3; display:flex; flex-direction:column;
  align-items:center; justify-content:center; color:#F2F2F2; font-size: 14px; gap: 14px; }
.av-spin { width: 26px; height: 26px; border: 2.5px solid rgba(255,255,255,.25); border-top-color:#fff;
  border-radius: 50%; animation: avspin 0.9s linear infinite; }
@keyframes avspin { to { transform: rotate(360deg); } }
div[class*="st-key-camframe"] img { border-radius: 18px; }
.av-foot { text-align:center; font-size: 12px; color:#9A9392; margin-top: .4rem; }

/* Resultado de traducción */
.av-result { border-radius: 16px; padding: 16px 20px; background:#FFF7F5; border: 1px solid #F8DCD6; }
.av-result .k { font-size: 11px; letter-spacing: .24em; color:#A0908D; }
.av-result .v { font-size: 34px; font-weight: 600; color:#C0645A; line-height: 1.2; margin-top: 4px; }
.av-result .m { font-size: 14px; color:#5E5553; margin-top: 2px; }
.av-conf { display:inline-block; font-size: 11px; padding: 2px 9px; border-radius: 999px; background:#fff;
  color:#8A6F6A; border: 1px solid #F1D5CF; margin-left: 8px; vertical-align: middle; }
.av-hist { font-size: 13px; color:#6F6866; margin-top: .6rem; }
.av-easy li { font-size: 22px; line-height: 1.6; color:#2F2A29; }
.av-ticket { border-radius: 16px; padding: 16px 20px; background:#F4FBF6; border:1px solid #CFEBD8; color:#2E5B3C; }
div.st-key-back button { background: none; border: 0; color:#A0908D; padding: 0; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

ICONS = {
    "mic": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
    "ear": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M7 9a5 5 0 0 1 10 0c0 3-3 4-3 7a3 3 0 0 1-6 0"/><path d="M10 10a2 2 0 0 1 4 0"/></svg>',
    "cam": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M4 8h3l2-2.5h6L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>',
}

# ───────────────────────── Estado y recursos ─────────────────────────
ss = st.session_state
ss.setdefault("mode", "inicio")
ss.setdefault("script", None)       # guion actual del avatar
ss.setdefault("history", [])        # [(hora, seña, confianza)]
ss.setdefault("last_sign", None)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(config.MODEL_PATH) if config.MODEL_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def get_tracker():
    from core.landmarks import HolisticTracker
    return HolisticTracker(config.MP_MODELS_DIR)


def respond(text: str, glosses: list[str]) -> dict:
    """Texto del banco -> audio + guion del avatar."""
    try:
        speech = synthesize(text)
    except Exception as exc:
        st.toast(f"Sin audio (continúa solo en señas): {exc}", icon="🔇")
        speech = None
    return build_script(text, glosses, speech)


def show_avatar(slot, script):
    slot.empty()
    with slot.container():
        render_avatar(script, height=640)


def go(mode: str):
    ss.mode = mode
    st.rerun()


def option_card(key: str, icon: str, title: str, desc: str, badge: bool = False) -> bool:
    with st.container(key=f"opt_{key}"):
        st.markdown(
            f'<div class="av-card"><div class="av-icon">{ICONS[icon]}</div>'
            f'<div><div class="av-title">{title}</div><div class="av-desc">{desc}</div></div>'
            f'{"<span class=av-badge>✦ AI</span>" if badge else ""}</div>', unsafe_allow_html=True)
        return st.button(title, key=f"btn_{key}")


def back_button():
    if st.button("← Volver a opciones", key="back"):
        go("inicio")


# ───────────────────────── Paneles ─────────────────────────
def panel_inicio():
    st.markdown('<div style="height:14vh"></div><div class="av-section">OPCIONES DE ASISTENCIA</div>',
                unsafe_allow_html=True)
    if option_card("voz", "mic", "Traducir voz a señas",
                   "El funcionario habla y el avatar anima la respuesta en la pantalla."):
        go("voz")
    if option_card("int", "ear", "Solicitar intérprete / lectura fácil",
                   "Intérprete humano o simplificación de textos por IA."):
        go("interprete")
    if option_card("cam", "cam", "Intérprete de señas (vía cámara)",
                   "Usa la cámara y Visión Artificial para personas con discapacidad del habla.", badge=True):
        go("camara")


def panel_voz(avatar_slot):
    back_button()
    st.markdown('<div class="av-section">TRADUCIR VOZ A SEÑAS</div>', unsafe_allow_html=True)
    quick = st.selectbox("Frases rápidas", ["— escribir mensaje —", *config.QUICK_PHRASES])
    text = st.text_area("Mensaje del funcionario", value="" if quick.startswith("—") else quick, height=110,
                        placeholder="Ej.: Por favor, espere un momento.")
    glosses = text_to_gloss(text)
    st.caption("Glosas LSRD: " + (" · ".join(glosses) if glosses else "— (sin señas en el léxico)"))
    if st.button("Signar y hablar", type="primary", disabled=not text.strip()):
        ss.script = respond(text.strip(), glosses)
        show_avatar(avatar_slot, ss.script)


def panel_interprete(avatar_slot):
    back_button()
    st.markdown('<div class="av-section">INTÉRPRETE HUMANO</div>', unsafe_allow_html=True)
    if st.button("Solicitar intérprete de LSRD", type="primary"):
        ss.ticket = f"I-{int(time.time()) % 1000:03d}"
    if "ticket" in ss:
        st.markdown(f'<div class="av-ticket">Intérprete solicitado · turno <b>{ss.ticket}</b> · '
                    f'tiempo estimado 3 min (simulado en el MVP)</div>', unsafe_allow_html=True)

    st.markdown('<div class="av-section">LECTURA FÁCIL</div>', unsafe_allow_html=True)
    topic = st.radio("Trámite", config.SIGNS, horizontal=True, format_func=lambda s: config.SIGN_LABELS[s])
    lines = config.EASY_READ[topic]
    st.markdown('<ul class="av-easy">' + "".join(f"<li>{l}</li>" for l in lines) + "</ul>", unsafe_allow_html=True)
    if st.button("Leer en voz alta y signar"):
        text = " ".join(lines)
        ss.script = respond(text, text_to_gloss(text))
        show_avatar(avatar_slot, ss.script)


def result_html(sign: str | None, conf: float | None) -> str:
    if not sign:
        return ('<div class="av-result"><div class="k">TRADUCCIÓN</div>'
                '<div class="m">Esperando una seña del cliente…</div></div>')
    hist = " · ".join(f"{t} {config.SIGN_LABELS[s]}" for t, s, _ in ss.history[-5:][::-1])
    return (f'<div class="av-result"><div class="k">EL CLIENTE DICE <span class="av-conf">{conf:.0%}</span></div>'
            f'<div class="v">{config.SIGN_LABELS[sign]}</div><div class="m">{config.SIGN_MEANING[sign]}</div></div>'
            f'<div class="av-hist">Historial: {hist}</div>')


def show_frame(slot, frame_bgr):
    try:
        slot.image(frame_bgr, channels="BGR", width="stretch")
    except TypeError:  # Streamlit < 1.45
        slot.image(frame_bgr, channels="BGR", use_container_width=True)


def panel_camara(avatar_slot):
    from core.landmarks import draw_landmarks, extract_frame_vector, sequence_to_features

    back_button()
    st.markdown('<div class="av-lead">Activa la cámara y la Visión Artificial detecta las señas o gestos del '
                'ciudadano y los traduce a texto para el funcionario.</div>', unsafe_allow_html=True)
    bundle = load_model()
    if bundle is None:
        st.warning("No hay modelo entrenado. Ejecuta `python 1_capture_data.py` y luego `python 2_train_model.py`.")
    elif bundle.get("backend") == "synthetic":
        st.warning("El modelo actual se entrenó con datos SINTÉTICOS (solo prueba de la interfaz): sus detecciones "
                   "con cámara real no son válidas. Captura señas reales con `1_capture_data.py` y reentrena.")
    c1, c2 = st.columns([1, 1])
    run = c1.toggle("Activar cámara", key="cam_on")
    cam_idx = c2.number_input("Cámara", 0, 5, 0, label_visibility="collapsed")

    with st.container(key="camframe"):
        frame_slot = st.empty()
    st.markdown('<div class="av-foot">📷 MediaPipe Holistic · procesamiento local en el dispositivo</div>',
                unsafe_allow_html=True)
    result_slot = st.empty()
    last = ss.history[-1] if ss.history else None
    result_slot.markdown(result_html(last[1] if last else None, last[2] if last else None), unsafe_allow_html=True)

    if not run:
        frame_slot.markdown(f'<div class="av-cam">{ICONS["cam"]}<div>Cámara apagada</div></div>',
                            unsafe_allow_html=True)
        return

    frame_slot.markdown('<div class="av-cam"><div class="av-spin"></div>'
                        '<div>Cargando modelo de Visión Artificial…</div></div>', unsafe_allow_html=True)
    tracker = get_tracker()
    if bundle and bundle.get("backend") not in (tracker.backend, "synthetic"):
        st.info(f"El modelo se entrenó con el backend '{bundle['backend']}' y la app usa '{tracker.backend}'. "
                "Para mejor precisión, captura y entrena con el mismo entorno.")

    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        frame_slot.error(f"No se pudo abrir la cámara {cam_idx}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    seq_len = bundle["seq_len"] if bundle else config.SEQ_LEN
    buf, probs = deque(maxlen=seq_len), deque(maxlen=config.SMOOTH_WINDOW)
    n, last_fire = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                frame_slot.error("La cámara dejó de enviar imágenes.")
                break
            frame = cv2.flip(frame, 1)                       # espejo, igual que en la captura
            lm = tracker.process(frame)
            buf.append(extract_frame_vector(lm))
            n += 1
            draw_landmarks(frame, lm)

            if bundle and len(buf) == seq_len and n % config.PREDICT_EVERY == 0:
                if lm.has_hand:
                    pipe = bundle["pipeline"]
                    probs.append(pipe.predict_proba(sequence_to_features(np.stack(buf))[None])[0])
                else:
                    probs.clear()                             # sin manos = sin seña
                if len(probs) >= config.SMOOTH_WINDOW // 2:
                    P = np.stack(probs)
                    votes = Counter(P.argmax(1))
                    top, count = votes.most_common(1)[0]
                    conf = float(P[:, top].mean())
                    now = time.monotonic()
                    if (conf >= config.CONF_THRESHOLD and count >= 0.6 * len(P)
                            and now - last_fire > config.COOLDOWN_S):
                        sign = bundle["labels"][int(bundle["pipeline"].classes_[top])]
                        last_fire = now
                        probs.clear()
                        ss.history.append((time.strftime("%H:%M:%S"), sign, conf))
                        result_slot.markdown(result_html(sign, conf), unsafe_allow_html=True)
                        if sign != ss.last_sign or now - ss.get("last_sign_t", 0) > 8:
                            r = config.BANK_RESPONSES[sign]
                            ss.script = respond(r["text"], r["gloss"])
                            show_avatar(avatar_slot, ss.script)
                        ss.last_sign, ss.last_sign_t = sign, now
            show_frame(frame_slot, frame)
    finally:
        cap.release()


# ───────────────────────── Layout ─────────────────────────
st.markdown('<div class="av-brand"><div class="av-logo">AccesoVivo</div>'
            '<div class="av-tag">INCLUSIÓN MULTIMODAL</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")
with left:
    avatar_slot = st.empty()
    show_avatar(avatar_slot, ss.script)
with right:
    {"inicio": lambda: panel_inicio(),
     "voz": lambda: panel_voz(avatar_slot),
     "interprete": lambda: panel_interprete(avatar_slot),
     "camara": lambda: panel_camara(avatar_slot)}[ss.mode]()
