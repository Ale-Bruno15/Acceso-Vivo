"""Extracción y normalización de landmarks con MediaPipe (backend doble).

Backend 1 ("solutions"): mp.solutions.holistic  -> mediapipe<=0.10.21 (Python <=3.12). Recomendado.
Backend 2 ("tasks"):     mediapipe.tasks HolisticLandmarker, o Hand+Pose+Face si no existe (mediapipe>=0.10.31).

Ambos producen el MISMO vector por frame, así que dataset, modelo y app son intercambiables.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Layout del vector por frame --------------------------------------------
N_POSE, N_HAND = 33, 21
# Subconjunto facial (índices FaceMesh): cejas, ojos, labios -> marcadores no manuales de LSRD
FACE_IDX = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,              # cejas
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # ojos
    61, 291, 0, 17, 13, 14, 78, 308, 81, 311, 178, 402,         # labios
    1, 4, 152, 10, 234, 454,                                    # nariz, mentón, frente, mejillas
]
N_FACE = len(FACE_IDX)

POSE_DIM = N_POSE * 4          # x, y, z, visibility
HAND_DIM = N_HAND * 3
FACE_DIM = N_FACE * 3
FRAME_DIM = POSE_DIM + 2 * HAND_DIM + FACE_DIM   # 132 + 126 + 120 = 378

SL_POSE = slice(0, POSE_DIM)
SL_LH = slice(POSE_DIM, POSE_DIM + HAND_DIM)
SL_RH = slice(POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM)
SL_FACE = slice(POSE_DIM + 2 * HAND_DIM, FRAME_DIM)

_MP = "https://storage.googleapis.com/mediapipe-models"
TASK_URLS = {
    "holistic": f"{_MP}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
    "hand": f"{_MP}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": f"{_MP}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face": f"{_MP}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


@dataclass
class Landmarks:
    """Resultado común a todos los backends, en coordenadas de imagen normalizadas [0, 1]."""
    pose: np.ndarray | None = None        # (33, 4)  x, y, z, visibility
    left_hand: np.ndarray | None = None   # (21, 3)
    right_hand: np.ndarray | None = None  # (21, 3)
    face: np.ndarray | None = None        # (468+, 3)
    extra: dict = field(default_factory=dict)

    @property
    def has_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None


def _arr(lms, with_vis: bool = False):
    if not lms:
        return None
    if with_vis:
        return np.array([[p.x, p.y, p.z, getattr(p, "visibility", None) or 0.0] for p in lms], np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms], np.float32)


def _download(kind: str, models_dir: Path) -> str:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / Path(TASK_URLS[kind]).name
    if not path.exists():
        print(f"[landmarks] Descargando modelo '{kind}' -> {path} ...")
        urllib.request.urlretrieve(TASK_URLS[kind], path)
    return str(path)


class HolisticTracker:
    """Wrapper único sobre MediaPipe.

    Uso:
        with HolisticTracker() as tracker:
            lm = tracker.process(frame_bgr)
    """

    def __init__(self, models_dir: Path | str = "models/mediapipe", min_conf: float = 0.5):
        import mediapipe as mp

        self._mp = mp
        self.models_dir = Path(models_dir)
        self.min_conf = min_conf
        self._t0 = time.monotonic()
        self._last_ts = -1
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "holistic"):
            self.backend = "solutions"
            self._holistic = solutions.holistic.Holistic(
                model_complexity=1, smooth_landmarks=True, refine_face_landmarks=False,
                min_detection_confidence=min_conf, min_tracking_confidence=min_conf,
            )
        else:
            self._init_tasks()

    # -- Tasks API ------------------------------------------------------------
    def _init_tasks(self):
        from mediapipe.tasks.python import BaseOptions, vision

        mode = vision.RunningMode.VIDEO
        if hasattr(vision, "HolisticLandmarker"):
            try:
                opts = vision.HolisticLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_download("holistic", self.models_dir)),
                    running_mode=mode,
                    min_face_detection_confidence=self.min_conf,
                    min_pose_detection_confidence=self.min_conf,
                    min_hand_landmarks_confidence=self.min_conf,
                )
                self._holistic = vision.HolisticLandmarker.create_from_options(opts)
                self.backend = "tasks-holistic"
                return
            except Exception as exc:  # la clase existe pero no funciona en algunas builds
                print(f"[landmarks] HolisticLandmarker no disponible ({exc}); usando Hand+Pose+Face.")

        self.backend = "tasks-combo"
        self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_download("hand", self.models_dir)),
            running_mode=mode, num_hands=2, min_hand_detection_confidence=self.min_conf))
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_download("pose", self.models_dir)),
            running_mode=mode, min_pose_detection_confidence=self.min_conf))
        self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_download("face", self.models_dir)),
            running_mode=mode, num_faces=1, min_face_detection_confidence=self.min_conf))

    def _timestamp_ms(self) -> int:
        ts = int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)  # el modo VIDEO exige timestamps estrictamente crecientes
        self._last_ts = ts
        return ts

    # -- API pública ----------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> Landmarks:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        if self.backend == "solutions":
            rgb.flags.writeable = False
            r = self._holistic.process(rgb)
            return Landmarks(
                pose=_arr(r.pose_landmarks.landmark if r.pose_landmarks else None, True),
                left_hand=_arr(r.left_hand_landmarks.landmark if r.left_hand_landmarks else None),
                right_hand=_arr(r.right_hand_landmarks.landmark if r.right_hand_landmarks else None),
                face=_arr(r.face_landmarks.landmark if r.face_landmarks else None),
            )

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms()
        if self.backend == "tasks-holistic":
            r = self._holistic.detect_for_video(image, ts)
            return Landmarks(
                pose=_arr(r.pose_landmarks, True),
                left_hand=_arr(r.left_hand_landmarks),
                right_hand=_arr(r.right_hand_landmarks),
                face=_arr(r.face_landmarks),
            )

        # tasks-combo
        hr = self._hand.detect_for_video(image, ts)
        pr = self._pose.detect_for_video(image, ts)
        fr = self._face.detect_for_video(image, ts)
        lm = Landmarks(
            pose=_arr(pr.pose_landmarks[0], True) if pr.pose_landmarks else None,
            face=_arr(fr.face_landmarks[0]) if fr.face_landmarks else None,
        )
        for hand, handed in zip(hr.hand_landmarks, hr.handedness):
            # Con la imagen espejada (modo selfie) la etiqueta coincide con la de mp.solutions.holistic.
            if handed[0].category_name == "Left":
                lm.left_hand = _arr(hand)
            else:
                lm.right_hand = _arr(hand)
        return lm

    def close(self):
        for name in ("_holistic", "_hand", "_pose", "_face"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- Vector por frame y features ---------------------------------------------
def extract_frame_vector(lm: Landmarks) -> np.ndarray:
    """Landmarks -> vector (FRAME_DIM,) normalizado respecto a los hombros. Partes ausentes = 0.

    Origen = punto medio de los hombros; escala = ancho de hombros. Así el vector es invariante
    a la distancia a la cámara y a la posición del cliente frente a la pantalla.
    """
    vec = np.zeros(FRAME_DIM, np.float32)

    if lm.pose is not None:
        l_sh, r_sh = lm.pose[11, :3], lm.pose[12, :3]
        origin = (l_sh + r_sh) / 2.0
        scale = float(np.linalg.norm(l_sh[:2] - r_sh[:2])) or 1.0
    else:  # sin pose: centro de imagen y escala nominal
        origin, scale = np.array([0.5, 0.5, 0.0], np.float32), 0.25

    def norm(pts):
        return (pts[:, :3] - origin) / scale

    if lm.pose is not None:
        p = lm.pose.copy()
        p[:, :3] = norm(p)
        vec[SL_POSE] = p.ravel()
    if lm.left_hand is not None:
        vec[SL_LH] = norm(lm.left_hand).ravel()
    if lm.right_hand is not None:
        vec[SL_RH] = norm(lm.right_hand).ravel()
    if lm.face is not None and len(lm.face) > max(FACE_IDX):
        vec[SL_FACE] = norm(lm.face[FACE_IDX]).ravel()
    return vec


N_STATS = 8
FEATURE_DIM = N_STATS * FRAME_DIM + 2   # + ratio de presencia de cada mano


def sequence_to_features(seq: np.ndarray) -> np.ndarray:
    """(T, FRAME_DIM) -> vector fijo (FEATURE_DIM,) con estadísticas temporales.

    media, std, min, max, primer frame, último frame, desplazamiento total y |velocidad| media:
    capturan forma de mano, ubicación y movimiento sin necesitar un modelo secuencial (LSTM).
    """
    seq = np.asarray(seq, np.float32)
    if seq.ndim != 2 or seq.shape[1] != FRAME_DIM:
        raise ValueError(f"Se esperaba (T, {FRAME_DIM}), llegó {seq.shape}")
    vel = np.abs(np.diff(seq, axis=0)) if len(seq) > 1 else np.zeros_like(seq)
    lh_present = np.any(seq[:, SL_LH] != 0, axis=1).mean()
    rh_present = np.any(seq[:, SL_RH] != 0, axis=1).mean()
    return np.concatenate([
        seq.mean(0), seq.std(0), seq.min(0), seq.max(0),
        seq[0], seq[-1], seq[-1] - seq[0], vel.mean(0),
        [lh_present, rh_present],
    ]).astype(np.float32)


def sequences_to_features(X: np.ndarray) -> np.ndarray:
    """(N, T, FRAME_DIM) -> (N, FEATURE_DIM)."""
    return np.stack([sequence_to_features(s) for s in X])


# --- Dibujo (independiente del backend) --------------------------------------
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]
_POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def draw_landmarks(frame_bgr: np.ndarray, lm: Landmarks) -> np.ndarray:
    """Dibuja torso, manos y subconjunto facial sobre el frame (in-place)."""
    import cv2

    h, w = frame_bgr.shape[:2]

    def px(p):
        return int(p[0] * w), int(p[1] * h)

    if lm.pose is not None:
        for a, b in _POSE_EDGES:
            if lm.pose[a, 3] > 0.5 and lm.pose[b, 3] > 0.5:
                cv2.line(frame_bgr, px(lm.pose[a]), px(lm.pose[b]), (200, 200, 200), 2, cv2.LINE_AA)
    for hand, color in ((lm.left_hand, (141, 153, 232)), (lm.right_hand, (180, 130, 255))):
        if hand is None:
            continue
        for a, b in _HAND_EDGES:
            cv2.line(frame_bgr, px(hand[a]), px(hand[b]), color, 2, cv2.LINE_AA)
        for p in hand:
            cv2.circle(frame_bgr, px(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if lm.face is not None and len(lm.face) > max(FACE_IDX):
        for i in FACE_IDX:
            cv2.circle(frame_bgr, px(lm.face[i]), 1, (220, 220, 220), -1, cv2.LINE_AA)
    return frame_bgr
