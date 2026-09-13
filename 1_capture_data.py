"""Script 1 — Captura de datos: graba secuencias de landmarks de señas bancarias con la webcam.

Uso:
    python 1_capture_data.py                       # deposito, retiro, ayuda — 30 secuencias c/u
    python 1_capture_data.py --sequences 40 --camera 1

Controles (ventana OpenCV):
    1 / 2 / 3 ...  seleccionar seña
    ESPACIO        iniciar / pausar grabación automática (cuenta regresiva entre muestras)
    BACKSPACE      borrar la última muestra de la seña actual
    Q / ESC        salir (consolida data/dataset.npz)

Cada muestra = SEQ_LEN frames (≈1 s) -> data/raw/<seña>/<n>.npy de forma (SEQ_LEN, FRAME_DIM).
Consejos: varía ligeramente distancia, iluminación y ropa; si puedes, graba a varias personas.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

import config
from core.landmarks import FRAME_DIM, HolisticTracker, draw_landmarks, extract_frame_vector

PINK = (141, 153, 232)   # BGR ≈ #E8998D
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
COUNTDOWN_S = 1.5
MIN_HAND_RATIO = 0.6     # descarta muestras donde casi no se vieron las manos


def sign_dir(sign: str):
    d = config.RAW_DIR / sign
    d.mkdir(parents=True, exist_ok=True)
    return d


def count(sign: str) -> int:
    return len(list(sign_dir(sign).glob("*.npy")))


def save_sequence(sign: str, seq: np.ndarray) -> int:
    existing = sorted(int(p.stem) for p in sign_dir(sign).glob("*.npy"))
    idx = (existing[-1] + 1) if existing else 0
    np.save(sign_dir(sign) / f"{idx:04d}.npy", seq)
    return idx


def delete_last(sign: str) -> bool:
    files = sorted(sign_dir(sign).glob("*.npy"))
    if files:
        files[-1].unlink()
        return True
    return False


def consolidate(signs: list[str], backend: str) -> None:
    X, y = [], []
    for label, sign in enumerate(signs):
        for f in sorted(sign_dir(sign).glob("*.npy")):
            seq = np.load(f)
            if seq.shape == (config.SEQ_LEN, FRAME_DIM):
                X.append(seq)
                y.append(label)
    if not X:
        print("[captura] No hay muestras que consolidar.")
        return
    np.savez_compressed(
        config.DATASET_PATH, X=np.stack(X).astype(np.float32), y=np.array(y),
        labels=np.array(signs), backend=backend, seq_len=config.SEQ_LEN,
    )
    per_class = {s: int((np.array(y) == i).sum()) for i, s in enumerate(signs)}
    print(f"[captura] Dataset -> {config.DATASET_PATH}  X={np.stack(X).shape}  {per_class}")


def put(img, text, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, DARK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(img, signs, current, target, state, progress, hand_ok, msg):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), DARK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    put(img, "AccesoVivo - captura de senas", (14, 26), 0.6)
    x = 14
    for i, s in enumerate(signs):
        n = count(s)
        label = f"[{i + 1}] {config.SIGN_LABELS.get(s, s)} {n}/{target}"
        color = PINK if i == current else ((170, 220, 170) if n >= target else WHITE)
        put(img, label, (x, 56), 0.55, color, 2 if i == current else 1)
        x += 14 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] + 16
    status = {"idle": "ESPACIO = grabar", "countdown": "Preparate...", "recording": "GRABANDO"}[state]
    put(img, f"{status}   |  manos: {'OK' if hand_ok else '--'}   |  {msg}", (14, 82), 0.5)
    if state == "recording":
        cv2.circle(img, (w - 26, 26), 10, (60, 60, 230), -1)
        cv2.rectangle(img, (0, h - 8), (int(w * progress), h), PINK, -1)
    elif state == "countdown":
        put(img, f"{max(0.0, progress):.1f}", (w // 2 - 30, h // 2), 2.2, PINK, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", nargs="+", default=config.SIGNS)
    ap.add_argument("--sequences", type=int, default=config.SEQUENCES_PER_SIGN)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, api)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.camera}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current, state, auto = 0, "idle", False
    t_state, buf, hand_frames, msg = 0.0, [], 0, "listo"

    with HolisticTracker(config.MP_MODELS_DIR) as tracker:
        print(f"[captura] Backend MediaPipe: {tracker.backend}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    msg = "sin frame de camara"
                    continue
                frame = cv2.flip(frame, 1)  # modo espejo (igual que en app.py)
                lm = tracker.process(frame)
                draw_landmarks(frame, lm)
                sign = args.signs[current]
                now = time.monotonic()

                if state == "countdown" and now - t_state >= COUNTDOWN_S:
                    state, buf, hand_frames = "recording", [], 0
                elif state == "recording":
                    buf.append(extract_frame_vector(lm))
                    hand_frames += int(lm.has_hand)
                    if len(buf) == config.SEQ_LEN:
                        if hand_frames / config.SEQ_LEN < MIN_HAND_RATIO:
                            msg = "muestra descartada: manos poco visibles"
                        else:
                            idx = save_sequence(sign, np.stack(buf))
                            msg = f"guardada {sign}/{idx:04d}"
                        if auto and count(sign) < args.sequences:
                            state, t_state = "countdown", now
                        else:
                            state, auto = "idle", False
                            if count(sign) >= args.sequences:
                                msg = f"{sign} completo"

                progress = (len(buf) / config.SEQ_LEN) if state == "recording" else COUNTDOWN_S - (now - t_state)
                draw_hud(frame, args.signs, current, args.sequences, state, progress, lm.has_hand, msg)
                cv2.imshow("AccesoVivo - Captura", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if state == "idle":
                        auto, state, t_state = True, "countdown", now
                    else:
                        auto, state, msg = False, "idle", "pausado"
                elif key == 8:  # backspace
                    msg = "ultima muestra borrada" if delete_last(sign) else "nada que borrar"
                elif ord("1") <= key <= ord("9") and key - ord("1") < len(args.signs) and state == "idle":
                    current = key - ord("1")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            consolidate(args.signs, tracker.backend)


if __name__ == "__main__":
    main()
