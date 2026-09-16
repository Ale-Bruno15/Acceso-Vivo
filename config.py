"""Configuración central de AccesoVivo: señas, rutas, umbrales y diálogo del banco."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Rutas -----------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_PATH = DATA_DIR / "dataset.npz"
LEXICON_PATH = DATA_DIR / "lexicon_lsrd.json"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "lsd_classifier.joblib"
MP_MODELS_DIR = MODELS_DIR / "mediapipe"
TTS_CACHE_DIR = ROOT / "cache" / "tts"
AVATAR_GLB = ROOT / "static" / "avatar.glb"
# Streamlit sirve ./static/ con prefijos distintos segun el entorno y ninguno acepta el del otro:
# local -> /app/static/...   Streamlit Cloud -> /~/+/app/static/...
# Se emiten ambos y el navegador usa el que responda (la ruta equivocada devuelve HTML, no el archivo).
STATIC_PREFIXES = ("/app/static", "/~/+/app/static")
AVATAR_GLB_URLS = [f"{prefix}/avatar.glb" for prefix in STATIC_PREFIXES]

# --- Señas del cliente (entrada) --------------------------------------------
SIGNS = ["deposito", "retiro", "ayuda"]
SIGN_LABELS = {  # texto que ve el funcionario
    "deposito": "Depósito",
    "retiro": "Retiro",
    "ayuda": "Ayuda",
}
SIGN_MEANING = {
    "deposito": "El cliente desea realizar un depósito.",
    "retiro": "El cliente desea realizar un retiro.",
    "ayuda": "El cliente solicita asistencia.",
}

# --- Captura / modelo --------------------------------------------------------
SEQ_LEN = 30             # frames por muestra (~1 s a 30 fps)
SEQUENCES_PER_SIGN = 30
FEATURE_VERSION = 1

# --- Inferencia en tiempo real ----------------------------------------------
PREDICT_EVERY = 5        # predecir cada N frames
SMOOTH_WINDOW = 8        # predicciones para votación por mayoría
CONF_THRESHOLD = 0.75    # confianza media mínima
COOLDOWN_S = 2.5         # evita disparos repetidos de la misma seña

# --- Respuestas del banco (salida: texto + audio + glosas LSRD) -------------
# Las glosas siguen el orden de LSRD (no el del español) y deben existir en lexicon_lsrd.json.
BANK_RESPONSES = {
    "deposito": {
        "text": "Con gusto. Para su depósito, tenga su cédula a mano y espere un momento, por favor.",
        "gloss": ["SI", "DEPOSITO", "CEDULA", "TU", "ESPERAR", "POR-FAVOR"],
    },
    "retiro": {
        "text": "Claro. Para su retiro, necesito su cédula o su tarjeta. Espere un momento, por favor.",
        "gloss": ["SI", "RETIRO", "CEDULA", "TARJETA", "ESPERAR", "POR-FAVOR"],
    },
    "ayuda": {
        "text": "Hola, estoy aquí para ayudarle. ¿Qué necesita?",
        "gloss": ["HOLA", "YO", "AYUDA", "TU", "QUE"],
    },
    "saludo": {
        "text": "Bienvenido a su banco. ¿En qué le puedo ayudar?",
        "gloss": ["HOLA", "BIENVENIDO", "AYUDA", "QUE"],
    },
}

# Frases rápidas para "Traducir voz a señas" (funcionario → cliente)
QUICK_PHRASES = [
    "Bienvenido a su banco. ¿En qué le puedo ayudar?",
    "Por favor, espere un momento.",
    "Necesito su cédula.",
    "Su depósito fue realizado. Gracias.",
    "Su retiro está listo. Gracias.",
]

# Español → glosa (búsqueda por palabra, sin tildes, minúsculas)
WORD_TO_GLOSS = {
    "hola": "HOLA", "bienvenido": "BIENVENIDO", "bienvenida": "BIENVENIDO",
    "ayuda": "AYUDA", "ayudar": "AYUDA", "ayudarle": "AYUDA",
    "deposito": "DEPOSITO", "depositar": "DEPOSITO",
    "retiro": "RETIRO", "retirar": "RETIRO",
    "cedula": "CEDULA", "tarjeta": "TARJETA",
    "espere": "ESPERAR", "esperar": "ESPERAR", "momento": "ESPERAR",
    "gracias": "GRACIAS", "si": "SI", "claro": "SI", "listo": "SI", "realizado": "SI",
    "favor": "POR-FAVOR", "que": "QUE", "yo": "YO", "su": "TU", "usted": "TU",
    "dinero": "DINERO", "banco": "BANCO",
}

# Lectura fácil (opción 2)
EASY_READ = {
    "deposito": ["Usted quiere depositar dinero.", "Tenga su cédula.", "Espere su turno."],
    "retiro": ["Usted quiere sacar dinero.", "Tenga su cédula o tarjeta.", "Espere su turno."],
    "ayuda": ["Usted pidió ayuda.", "Una persona del banco viene.", "Espere aquí."],
}
