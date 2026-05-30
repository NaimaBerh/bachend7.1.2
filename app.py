# =============================================================================
#  FakeGuard / FPD - Backend Flask (Python)
#  Version 7.2.0 — Correction des FAUX POSITIFS systématiques
# =============================================================================
#
#  Correctifs v7.2.0 (vs v7.1.1) — RÉSOLUTION DES RÉSULTATS FAUX :
#  --------------------------------------------------------------
#    Diagnostic du bug :
#      Quand l'extraction d'URL échouait (Instagram/LinkedIn bloqués sans
#      cookies, X qui ne renvoie rien sans JS, etc.), le backend renvoyait
#      un profil rempli de zéros (followers=0, posts=0, has_pic=False, bio="").
#      Ces zéros étaient ensuite passés au modèle Random Forest, qui les
#      interprétait comme un profil bot ("fake") avec une confiance de 90-95 %.
#      Résultat : presque TOUS les profils analysés étaient déclarés "fake",
#      même si la précision/recall/F1 du modèle restaient bons sur leur
#      jeu de test d'origine (où les features étaient TOUJOURS valides).
#
#    Correctifs appliqués :
#      1) DATA-QUALITY GATE : avant toute inférence ML, on vérifie qu'on
#         a au moins quelques signaux non-nuls. Sinon, on renvoie un label
#         "insufficient_data" (et non plus "fake") avec risk_score=null.
#      2) HYBRIDATION ML + HEURISTIQUES : la probabilité finale combine
#         la sortie du RF (50 %) et un score heuristique fondé sur des
#         règles métier observables (50 %). Cela compense les biais du
#         dataset d'entraînement (centré sur petits profils privés).
#      3) SEUIL DE DÉCISION RÉHAUSSÉ : on passe de 0.5 à 0.65 pour le label
#         "fake", et on ajoute une zone grise "suspicious" entre 0.45-0.65.
#         Cela réduit considérablement les faux positifs.
#      4) LINKEDIN : exige >= 25 mots dans le blob TF-IDF, sinon
#         renvoie "insufficient_data" (sans LinkedIn, le modèle prédisait
#         "fake" à 64 % sur du texte vide).
#      5) GITHUB / BOTHAWK : la sortie RF est désormais utilisée comme
#         signal secondaire ; le score principal vient des heuristiques
#         observables (repos, commits, bio, email, ratio followers).
#      6) INSTAGRAM/TWITTER : si followers==0 ET posts==0 ET bio vide,
#         on renvoie "insufficient_data" même si le username a été extrait.
#
#  Correctifs v7.1.1 (vs v7.1.0) :
#  --------------------------------
#    * /api/analyze-url et /api/extract-url EXPOSENT DÉSORMAIS de manière
#      garantie les compteurs canoniques `follower_count`, `following_count`
#      et `post_count` :
#         - dans `features` (toutes plateformes),
#         - dans `extracted_profile` / `profile`,
#         - au plus haut niveau de la réponse JSON,
#         - + alias camelCase (`userFollowerCount`, `userFollowingCount`,
#           `userMediaCount`) pour compat front Instagram-style.
#      Auparavant, ces compteurs étaient écrasés par le `**res["features"]`
#      (LinkedIn renvoyait un text_blob, Instagram renvoyait du camelCase),
#      ce qui faisait disparaître les nombres côté UI lorsqu'on collait un
#      lien de profil.
#
#  Correctifs v7.1 (vs v7.0) :
#  ---------------------------
#    * INSTAGRAM : utilisation de l'endpoint public web_profile_info (avec
#      en-tête X-IG-App-ID) qui renvoie un JSON complet et stable
#      (followers/following/posts/bio/avatar/is_private/is_verified...).
#      Fallback OG + 3 regex multilingues (FR/EN) + body-scan JSON intégré.
#    * TWITTER / X : utilisation de cdn.syndication.twimg.com (JSON public,
#      sans authentification) en première intention, puis fallback Nitter,
#      puis OG.
#    * LINKEDIN : oEmbed officiel + extraction des microdonnées HTML
#      (itemprop="*") + scan JSON intégré code-pushed dans le HTML.
#    * GITHUB : enrichissement (orgs, langages, repos starred, commits
#      réels via Search API best-effort).
#    * JOBS : extraction microdata HTML (itemprop="*") en complément du
#      JSON-LD + multi-User-Agent.
#    * HTTP : rotation User-Agent, retries avec headers réalistes,
#      gestion des 403/429, suivi de redirection JS-meta-refresh.
#    * Tous les extracteurs renvoient maintenant un champ
#      "extraction_status" décrivant la méthode et la qualité de la collecte.
#
#  Modèles actifs (inchangés) :
#    - rf_instagram, scaler_instagram   (Instagram + Twitter/X)
#    - rf_linkedin, tfidf_linkedin       (LinkedIn)
#    - rf_bothawk, tfidf_bothawk         (GitHub - Bothawk)
#    - fake_job_lstm_model.tflite        (Offres d'emploi)
#
#  Endpoints publics (inchangés) :
#    GET  /                    -> infos API
#    GET  /health              -> healthcheck
#    GET  /api/models          -> liste des modèles ML chargés
#    POST /api/analyze         -> analyse d'un profil (features manuelles)
#    POST /api/analyze-job     -> analyse LSTM d'une offre d'emploi (texte)
#    POST /api/extract-url     -> extraction brute d'un profil/offre depuis URL
#    POST /api/analyze-url     -> extraction + scoring complet depuis URL
# =============================================================================

import os
import re
import json
import time
import random
import ipaddress
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote

# Réduction du bruit TensorFlow (avant import)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------------------------------------------------------- #
#  Configuration                                                                 #
# ----------------------------------------------------------------------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Modèles Random Forest spécialisés par plateforme
RF_INSTAGRAM_PATH     = os.path.join(MODELS_DIR, "rf_instagram.pkl")
SCALER_INSTAGRAM_PATH = os.path.join(MODELS_DIR, "scaler_instagram.pkl")

RF_LINKEDIN_PATH      = os.path.join(MODELS_DIR, "rf_linkedin.pkl")
TFIDF_LINKEDIN_PATH   = os.path.join(MODELS_DIR, "tfidf_linkedin.pkl")

RF_BOTHAWK_PATH       = os.path.join(MODELS_DIR, "rf_bothawk.pkl")
TFIDF_BOTHAWK_PATH    = os.path.join(MODELS_DIR, "tfidf_bothawk.pkl")

# Modèle LSTM offre d'emploi
LSTM_MODEL_PATH       = os.path.join(MODELS_DIR, "fake_job_lstm_model.tflite")
LSTM_TOKENIZER_PATH   = os.path.join(MODELS_DIR, "tokenizer.json")
LSTM_MAX_SEQUENCE_LEN = 200
LSTM_THRESHOLD        = float(os.environ.get("LSTM_THRESHOLD", "0.5"))

# Plateformes
SUPPORTED_PLATFORMS = {"instagram", "linkedin", "github", "twitter"}

# Mapping plateforme -> modèle utilisé pour le scoring
PLATFORM_MODEL_MAP = {
    "instagram": "rf_instagram",
    "twitter":   "rf_instagram",
    "linkedin":  "rf_linkedin",
    "github":    "rf_bothawk",
}

# CORS
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# Port
PORT = int(os.environ.get("PORT", "5000"))

# Sécurité réseau pour l'extraction d'URL
HTTP_TIMEOUT          = int(os.environ.get("HTTP_TIMEOUT", "15"))
HTTP_MAX_REDIRECTS    = 5
HTTP_MAX_BYTES        = 4 * 1024 * 1024   # 4 Mo de page max
HTTP_MAX_RETRIES      = 2                  # retries avec User-Agent différent

# Plusieurs User-Agents réalistes (rotation pour contourner les blocages basiques)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

# Token GitHub optionnel (60 -> 5000 req/h)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()

# Instances Nitter publiques (fallback Twitter). Aucune n'est garantie en ligne,
# on essaye en cascade.
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.kavin.rocks",
]

# Performances déclarées des modèles
MODEL_PERFORMANCE: Dict[str, Dict[str, Any]] = {
    "rf_instagram": {
        "name": "Random Forest Instagram",
        "platforms": ["instagram", "twitter"],
        "precision": 0.94, "recall": 0.92, "f1_score": 0.93, "roc_auc": 0.97,
    },
    "rf_linkedin": {
        "name": "Random Forest LinkedIn (TF-IDF)",
        "platforms": ["linkedin"],
        "precision": 0.91, "recall": 0.89, "f1_score": 0.90, "roc_auc": 0.95,
    },
    "rf_bothawk": {
        "name": "Random Forest GitHub (Bothawk)",
        "platforms": ["github"],
        "precision": 0.93, "recall": 0.90, "f1_score": 0.91, "roc_auc": 0.96,
    },
    "job_text_lstm": {
        "name": "LSTM Offres d'emploi",
        "platforms": ["job"],
        "precision": 0.96, "recall": 0.94, "f1_score": 0.95, "roc_auc": 0.98,
    },
}

# ----------------------------------------------------------------------------- #
#  App Flask                                                                     #
# ----------------------------------------------------------------------------- #

app = Flask(__name__)

if ALLOWED_ORIGINS == ["*"]:
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
else:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)


# ----------------------------------------------------------------------------- #
#  Chargement paresseux des modèles                                              #
# ----------------------------------------------------------------------------- #

_models: Dict[str, Any] = {}
_load_errors: Dict[str, str] = {}


def _safe_load(name: str, path: str) -> None:
    """Charge un objet joblib de manière sûre, mémorise l'erreur sinon."""
    if name in _models or name in _load_errors:
        return
    try:
        import joblib  # import différé
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _models[name] = joblib.load(path)
        print(f"[Models] OK -> {name} ({os.path.basename(path)})", flush=True)
    except Exception as exc:  # noqa: BLE001
        _load_errors[name] = f"{type(exc).__name__}: {exc}"
        print(f"[Models] ECHEC -> {name} : {_load_errors[name]}", flush=True)


def load_all_models() -> None:
    """Charge tous les modèles RF + scaler/tfidf en mémoire (idempotent)."""
    _safe_load("rf_instagram",      RF_INSTAGRAM_PATH)
    _safe_load("scaler_instagram",  SCALER_INSTAGRAM_PATH)
    _safe_load("rf_linkedin",       RF_LINKEDIN_PATH)
    _safe_load("tfidf_linkedin",    TFIDF_LINKEDIN_PATH)
    _safe_load("rf_bothawk",        RF_BOTHAWK_PATH)
    _safe_load("tfidf_bothawk",     TFIDF_BOTHAWK_PATH)


# ---- LSTM (TensorFlow) chargé séparément --------------------------------------

_lstm_interpreter = None
_lstm_tokenizer = None
_lstm_in = None
_lstm_out = None
_lstm_error: Optional[str] = None


# ---------------------------------------------------------------------------- #
#  Tokenizer Keras léger (pure Python) — évite la dépendance TensorFlow        #
# ---------------------------------------------------------------------------- #
#  Réplique exactement le comportement de tf.keras.preprocessing.text.Tokenizer
#  pour les opérations dont nous avons besoin (texts_to_sequences) à partir du
#  JSON exporté par Keras. Permet de remplacer 'tensorflow' (~500 Mo) par
#  'tflite-runtime' (~3 Mo) — crucial pour l'hébergement Render.
# ---------------------------------------------------------------------------- #

class _LiteKerasTokenizer:
    """Implémentation minimale compatible Keras Tokenizer (texts_to_sequences)."""

    def __init__(self, word_index: Dict[str, int], num_words: Optional[int],
                 filters: str, lower: bool, split: str,
                 oov_token: Optional[str]) -> None:
        self.word_index = word_index or {}
        self.num_words = num_words
        self.filters = filters or ""
        self.lower = bool(lower)
        self.split = split or " "
        self.oov_token = oov_token
        # Table de translation pour filtrer les caractères de ponctuation
        # exactement comme Keras (chaque caractère du filtre est remplacé par un espace)
        self._translation = str.maketrans(self.filters, self.split * len(self.filters))

    @classmethod
    def from_json(cls, json_text: str) -> "_LiteKerasTokenizer":
        data = json.loads(json_text)
        cfg = data.get("config", data)
        word_index_raw = cfg.get("word_index")
        if isinstance(word_index_raw, str):
            word_index = json.loads(word_index_raw)
        else:
            word_index = word_index_raw or {}
        # Indices peuvent être stockés en string -> int
        word_index = {str(k): int(v) for k, v in word_index.items()}
        return cls(
            word_index=word_index,
            num_words=cfg.get("num_words"),
            filters=cfg.get("filters",
                            '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
            lower=cfg.get("lower", True),
            split=cfg.get("split", " "),
            oov_token=cfg.get("oov_token"),
        )

    def _text_to_words(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        text = text.translate(self._translation)
        return [w for w in text.split(self.split) if w]

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        oov_idx = self.word_index.get(self.oov_token) if self.oov_token else None
        sequences: List[List[int]] = []
        for text in texts:
            seq: List[int] = []
            for word in self._text_to_words(text):
                idx = self.word_index.get(word)
                if idx is not None:
                    if self.num_words is None or idx < self.num_words:
                        seq.append(idx)
                    elif oov_idx is not None:
                        seq.append(oov_idx)
                elif oov_idx is not None:
                    seq.append(oov_idx)
            sequences.append(seq)
        return sequences


def _lite_pad_sequences(sequences: List[List[int]], maxlen: int,
                       dtype: str = "float32",
                       padding: str = "pre", truncating: str = "pre",
                       value: int = 0) -> np.ndarray:
    """Réplique tf.keras.preprocessing.sequence.pad_sequences (modes 'pre')."""
    out = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        s = list(seq)
        if len(s) > maxlen:
            if truncating == "pre":
                s = s[-maxlen:]
            else:
                s = s[:maxlen]
        if padding == "pre":
            out[i, -len(s):] = s
        else:
            out[i, :len(s)] = s
    return out


def _load_tflite_interpreter(model_path: str):
    """
    Charge un interpréteur TFLite en privilégiant 'tflite-runtime' (léger),
    avec repli sur 'tensorflow' si présent (compat. dev local).
    """
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=model_path)
    except ImportError:
        try:
            import tensorflow as tf  # type: ignore
            return tf.lite.Interpreter(model_path=model_path)
        except ImportError as exc:
            raise ImportError(
                "Ni 'tflite-runtime' ni 'tensorflow' ne sont installés. "
                "Ajoutez 'tflite-runtime' à requirements.txt."
            ) from exc


def load_lstm() -> None:
    """Charge le modèle TFLite + tokenizer Keras pour l'analyse d'offres."""
    global _lstm_interpreter, _lstm_tokenizer, _lstm_in, _lstm_out, _lstm_error
    if _lstm_interpreter is not None or _lstm_error is not None:
        return
    try:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(LSTM_MODEL_PATH)
        if not os.path.exists(LSTM_TOKENIZER_PATH):
            raise FileNotFoundError(LSTM_TOKENIZER_PATH)

        with open(LSTM_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            _lstm_tokenizer = _LiteKerasTokenizer.from_json(f.read())
        _lstm_interpreter = _load_tflite_interpreter(LSTM_MODEL_PATH)
        _lstm_interpreter.allocate_tensors()
        _lstm_in = _lstm_interpreter.get_input_details()
        _lstm_out = _lstm_interpreter.get_output_details()
        print("[LSTM] Modèle TFLite + tokenizer (lite) chargés.", flush=True)
    except Exception as exc:  # noqa: BLE001
        _lstm_error = f"{type(exc).__name__}: {exc}"
        print(f"[LSTM] Chargement impossible : {_lstm_error}", flush=True)


# ----------------------------------------------------------------------------- #
#  Helpers de typage                                                             #
# ----------------------------------------------------------------------------- #

def _safe_int(value: Any, default: int = 0) -> int:
    """Convertit en int avec support des suffixes K/M/B (1.2K, 3M, ...)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return default
    try:
        txt = str(value).strip().replace(",", "").replace("\xa0", "").replace(" ", "")
        if not txt:
            return default
        mult = 1
        if txt[-1].upper() == "K":
            mult, txt = 1_000, txt[:-1]
        elif txt[-1].upper() == "M":
            mult, txt = 1_000_000, txt[:-1]
        elif txt[-1].upper() == "B":
            mult, txt = 1_000_000_000, txt[:-1]
        return int(float(txt) * mult)
    except (ValueError, TypeError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "y", "oui")


def _count_digits(text: str) -> int:
    return sum(1 for c in str(text or "") if c.isdigit())


def _clean_text(text: Optional[str], limit: int = 600) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned[:limit]


def _coalesce(*values: Any) -> Any:
    """Retourne la première valeur non vide/non nulle."""
    for v in values:
        if v not in (None, "", 0, [], {}):
            return v
    return None


# ----------------------------------------------------------------------------- #
#  Sécurité SSRF et HTTP client robuste                                          #
# ----------------------------------------------------------------------------- #

# Plages d'IP interdites pour l'extraction d'URL (anti-SSRF)
_FORBIDDEN_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("100.64.0.0/10"),    # CGNAT
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_safe_public_url(url: str) -> Tuple[bool, str]:
    """Vérifie qu'une URL est publique, http(s), et ne pointe pas vers du réseau interne."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL invalide"
    if parsed.scheme not in ("http", "https"):
        return False, "Seuls http/https sont autorisés"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "Hôte manquant"
    if host == "localhost" or host.endswith(".local") or host.endswith(".internal"):
        return False, "Hôte interne refusé"
    # Résolution DNS + check IP
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False, "Résolution DNS impossible"
    for info in infos:
        ip_str = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if any(ip_obj in net for net in _FORBIDDEN_NETS):
            return False, f"IP interne refusée ({ip_str})"
        if ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified:
            return False, "IP non publique refusée"
    return True, "ok"


def _build_headers(user_agent: str, accept_json: bool = False,
                   referer: Optional[str] = None,
                   extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Construit un jeu d'en-têtes HTTP réalistes (Chrome-like)."""
    if accept_json:
        accept = "application/json, text/javascript, */*;q=0.9"
    else:
        accept = ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "image/avif,image/webp,*/*;q=0.8")
    headers = {
        "User-Agent": user_agent,
        "Accept": accept,
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if referer:
        headers["Referer"] = referer
        headers["Sec-Fetch-Site"] = "same-origin"
    if extra:
        headers.update(extra)
    return headers


def http_safe_get(
    url: str,
    extra_headers: Optional[Dict[str, str]] = None,
    accept_json: bool = False,
    referer: Optional[str] = None,
    allow_redirects: bool = True,
    expected_status: Optional[List[int]] = None,
) -> Optional[requests.Response]:
    """
    GET sécurisé : SSRF guard, timeouts stricts, limite de taille, retries
    avec rotation User-Agent. Renvoie la réponse même en 4xx si elle contient
    du contenu exploitable.
    """
    ok, reason = _is_safe_public_url(url)
    if not ok:
        raise ValueError(f"URL refusée : {reason}")

    expected_status = expected_status or [200, 201, 202, 203, 204, 206]
    last_exc: Optional[Exception] = None
    last_resp: Optional[requests.Response] = None

    # Essayer plusieurs User-Agents si la première tentative échoue (403/429)
    for attempt in range(HTTP_MAX_RETRIES + 1):
        ua = USER_AGENTS[attempt % len(USER_AGENTS)]
        headers = _build_headers(ua, accept_json=accept_json,
                                 referer=referer, extra=extra_headers)
        try:
            session = requests.Session()
            session.max_redirects = HTTP_MAX_REDIRECTS
            session.headers.update(headers)
            resp = session.get(url, timeout=HTTP_TIMEOUT,
                               allow_redirects=allow_redirects, stream=True)

            # Lecture bornée
            content = b""
            for chunk in resp.iter_content(chunk_size=16_384):
                if not chunk:
                    break
                content += chunk
                if len(content) >= HTTP_MAX_BYTES:
                    break
            resp._content = content  # type: ignore[attr-defined]
            last_resp = resp

            # Si la réponse est exploitable, on s'arrête
            if resp.status_code in expected_status:
                return resp
            # 4xx mais possiblement encore du HTML utile -> on continue les retries
            if resp.status_code in (403, 429) and attempt < HTTP_MAX_RETRIES:
                time.sleep(0.4 + random.random() * 0.4)
                continue
            # 404 / 410 etc. -> inutile de retenter
            if resp.status_code in (404, 410):
                return resp
            # On retourne quand même la dernière réponse (peut contenir du HTML utile)
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            print(f"[HTTP] tentative {attempt+1} sur {url} : {exc}", flush=True)
            time.sleep(0.3)
            continue

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        print(f"[HTTP] échec définitif sur {url} : {last_exc}", flush=True)
    return None


def http_safe_json(url: str, headers: Optional[Dict[str, str]] = None,
                   referer: Optional[str] = None) -> Optional[Any]:
    """Récupère un JSON depuis une URL publique avec les mêmes garanties que http_safe_get."""
    resp = http_safe_get(url, extra_headers=headers, accept_json=True, referer=referer)
    if resp is None:
        return None
    if resp.status_code >= 400:
        return None
    try:
        return resp.json()
    except (ValueError, json.JSONDecodeError):
        try:
            return json.loads(resp.text)
        except Exception:
            return None


# ----------------------------------------------------------------------------- #
#  Extraction OG / JSON-LD / Microdata générique                                 #
# ----------------------------------------------------------------------------- #

def _extract_og_meta(soup: BeautifulSoup) -> Dict[str, str]:
    """Récupère toutes les balises meta OpenGraph/Twitter/description/keywords."""
    meta: Dict[str, str] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name") or tag.get("itemprop")
        if not key:
            continue
        key = key.lower()
        val = tag.get("content")
        if not val:
            continue
        val = val.strip()
        if not val:
            continue
        if key.startswith(("og:", "twitter:", "article:", "profile:", "al:")):
            meta[key] = val
        elif key in ("description", "keywords", "author", "title"):
            meta.setdefault(key, val)
    # <title>
    if soup.title and soup.title.string:
        meta.setdefault("title", soup.title.string.strip())
    return meta


def _extract_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extrait tous les blocs JSON-LD (utile pour JobPosting, Person, Organization...)."""
    results: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or script.get_text() or "")
        except (ValueError, TypeError):
            continue
        if isinstance(payload, list):
            for p in payload:
                if isinstance(p, dict):
                    if "@graph" in p and isinstance(p["@graph"], list):
                        results.extend(g for g in p["@graph"] if isinstance(g, dict))
                    else:
                        results.append(p)
        elif isinstance(payload, dict):
            if "@graph" in payload and isinstance(payload["@graph"], list):
                results.extend(g for g in payload["@graph"] if isinstance(g, dict))
            else:
                results.append(payload)
    return results


def _extract_microdata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extrait les microdonnées HTML5 (itemprop) sous forme de dict aplati."""
    data: Dict[str, Any] = {}
    for el in soup.find_all(attrs={"itemprop": True}):
        prop = el.get("itemprop")
        if not prop:
            continue
        # Valeur : content > datetime > href > src > texte
        val = (el.get("content") or el.get("datetime")
               or el.get("href") or el.get("src") or el.get_text(" ", strip=True))
        if not val:
            continue
        val = str(val).strip()
        if not val:
            continue
        # Si la clé existe déjà, on transforme en liste
        if prop in data:
            existing = data[prop]
            if isinstance(existing, list):
                existing.append(val)
            else:
                data[prop] = [existing, val]
        else:
            data[prop] = val
    return data


def _find_in_html_json(html: str, *keys: str) -> Optional[Any]:
    """
    Recherche brute-force d'une clé JSON dans le HTML (pour les SPA qui
    inlinent leur état initial dans le HTML : Instagram, LinkedIn, X...).
    Retourne la première valeur trouvée pour l'une des clés.
    """
    for key in keys:
        # patterns possibles : "key":123  ou  "key":"val"  ou  \"key\":123
        for pat in (
            rf'"{re.escape(key)}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
            rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)',
            rf'"{re.escape(key)}"\s*:\s*(true|false)',
            rf'\\"{re.escape(key)}\\"\s*:\s*\\"([^"\\]*(?:\\.[^"\\]*)*)\\"',
            rf'\\"{re.escape(key)}\\"\s*:\s*(-?\d+(?:\.\d+)?)',
        ):
            m = re.search(pat, html)
            if m:
                v = m.group(1)
                if v in ("true", "false"):
                    return v == "true"
                try:
                    return int(v) if "." not in v else float(v)
                except (ValueError, TypeError):
                    return v
    return None


# ----------------------------------------------------------------------------- #
#  Détection de plateforme                                                       #
# ----------------------------------------------------------------------------- #

JOB_URL_HINTS = [
    "/jobs/", "/job/", "jobposting", "job-posting", "careers", "carriere",
    "emploi", "offre-emploi", "offres-emploi", "recrutement", "vacancy",
    "vacancies", "hiring", "apply", "postuler", "viewjob",
]
JOB_DOMAINS = {
    "indeed.com", "linkedin.com/jobs", "glassdoor.com", "monster.com",
    "pole-emploi.fr", "francetravail.fr", "welcometothejungle.com",
    "jobteaser.com", "apec.fr", "hellowork.com", "stepstone.fr",
    "regionsjob.com", "leboncoin.fr/emploi",
}


def detect_platform(url: str) -> Tuple[str, str]:
    """Renvoie (plateforme, méthode_de_detection)."""
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
    except ValueError:
        return ("generic", "invalid_url")
    host = (parsed.netloc or "").lower().replace("www.", "")
    path = (parsed.path or "").lower()
    full = host + path

    for jd in JOB_DOMAINS:
        if jd in full:
            return ("job", "job_domain")
    if any(h in path for h in JOB_URL_HINTS):
        return ("job", "job_path_hint")
    if "instagram.com" in host:
        return ("instagram", "domain")
    if host == "x.com" or host.endswith(".x.com") or "twitter.com" in host:
        return ("twitter", "domain")
    if "linkedin.com" in host:
        return ("linkedin", "domain")
    if "github.com" in host:
        return ("github", "domain")
    return ("generic", "fallback")


# ----------------------------------------------------------------------------- #
#  EXTRACTEUR INSTAGRAM                                                          #
# ----------------------------------------------------------------------------- #
#
# Stratégie :
#   1) Endpoint public "web_profile_info" avec l'en-tête X-IG-App-ID=936619743392459
#      -> renvoie un JSON complet (followers, following, posts, bio, ...).
#   2) Fallback HTML : balises OG + recherche de clés JSON inlinées + 3 regex
#      multilingues (EN/FR) sur og:description.
#   3) Fallback final : juste le username extrait de l'URL.
# ----------------------------------------------------------------------------- #

INSTAGRAM_APP_ID = "936619743392459"
INSTAGRAM_REGEXES = [
    # EN : "1,234 Followers, 567 Following, 89 Posts"
    re.compile(
        r"([\d.,KMBkmb\xa0\s]+)\s*Followers?,?\s*"
        r"([\d.,KMBkmb\xa0\s]+)\s*Following,?\s*"
        r"([\d.,KMBkmb\xa0\s]+)\s*(?:Posts?|Publications?)",
        re.IGNORECASE,
    ),
    # FR : "1 234 abonnés, 567 abonnements, 89 publications"
    re.compile(
        r"([\d.,KMBkmb\xa0\s]+)\s*abonn[eé]s?,?\s*"
        r"([\d.,KMBkmb\xa0\s]+)\s*abonnements?,?\s*"
        r"([\d.,KMBkmb\xa0\s]+)\s*publications?",
        re.IGNORECASE,
    ),
    # ES : "Seguidores ... Siguiendo ... publicaciones"
    re.compile(
        r"([\d.,KMBkmb\xa0\s]+)\s*Seguidores?,?\s*"
        r"([\d.,KMBkmb\xa0\s]+)\s*Siguiendo,?\s*"
        r"([\d.,KMBkmb\xa0\s]+)\s*publicaciones?",
        re.IGNORECASE,
    ),
]


def _instagram_username_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in (parsed.path or "").split("/") if p]
    # /username/   ou   /username/p/xxx/
    for p in parts:
        if p not in ("explore", "p", "reel", "tv", "stories", "accounts"):
            return p
    return ""


def _instagram_via_api(username: str) -> Optional[Dict[str, Any]]:
    """Endpoint public web_profile_info (renvoie un JSON exploitable)."""
    if not username:
        return None
    api_url = f"https://i.instagram.com/api/v1/users/web_profile_info/?username={quote(username)}"
    headers = {
        "X-IG-App-ID": INSTAGRAM_APP_ID,
        "X-ASBD-ID": "129477",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.instagram.com",
    }
    payload = http_safe_json(api_url, headers=headers,
                             referer=f"https://www.instagram.com/{username}/")
    if not payload or not isinstance(payload, dict):
        return None
    user = (payload.get("data") or {}).get("user") or {}
    if not user:
        return None

    followers = _safe_int((user.get("edge_followed_by") or {}).get("count"))
    following = _safe_int((user.get("edge_follow") or {}).get("count"))
    posts = _safe_int((user.get("edge_owner_to_timeline_media") or {}).get("count"))
    return {
        "username":        user.get("username") or username,
        "display_name":    user.get("full_name") or user.get("username") or username,
        "bio":             _clean_text(user.get("biography"), 1500),
        "avatar_url":      user.get("profile_pic_url_hd") or user.get("profile_pic_url"),
        "profile_url":     f"https://www.instagram.com/{user.get('username', username)}/",
        "follower_count":  followers,
        "following_count": following,
        "post_count":      posts,
        "is_private":      bool(user.get("is_private", False)),
        "is_verified":     bool(user.get("is_verified", False)),
        "is_business":     bool(user.get("is_business_account", False)),
        "external_url":    user.get("external_url") or "",
        "category":        user.get("category_name") or user.get("business_category_name") or "",
        "account_type":    "instagram_user",
        "extraction_method": "instagram_web_profile_info_api",
    }


def _instagram_via_html(url: str, username: str) -> Optional[Dict[str, Any]]:
    """Fallback HTML : OG + regex multilingues + scan JSON inliné."""
    r = http_safe_get(url, referer="https://www.instagram.com/")
    if r is None or r.status_code >= 400:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    og_desc = meta.get("og:description") or meta.get("description") or ""

    followers = following = posts = 0
    for rx in INSTAGRAM_REGEXES:
        m = rx.search(og_desc)
        if m:
            followers = _safe_int(m.group(1))
            following = _safe_int(m.group(2))
            posts = _safe_int(m.group(3))
            break

    # Scan complémentaire des clés JSON inlinées (état initial Instagram)
    html = r.text
    if followers == 0:
        v = _find_in_html_json(html, "edge_followed_by", "follower_count")
        if isinstance(v, dict):
            v = v.get("count")
        if v is not None:
            followers = _safe_int(v)
    if following == 0:
        v = _find_in_html_json(html, "edge_follow", "following_count")
        if isinstance(v, dict):
            v = v.get("count")
        if v is not None:
            following = _safe_int(v)
    if posts == 0:
        v = _find_in_html_json(html, "edge_owner_to_timeline_media", "media_count")
        if isinstance(v, dict):
            v = v.get("count")
        if v is not None:
            posts = _safe_int(v)

    is_private = bool(_find_in_html_json(html, "is_private")) or False
    is_verified = bool(_find_in_html_json(html, "is_verified")) or False
    bio_inline = _find_in_html_json(html, "biography")
    full_name = _find_in_html_json(html, "full_name")

    bio_text = bio_inline if isinstance(bio_inline, str) and bio_inline else (
        og_desc.split(" - ")[-1] if " - " in og_desc else og_desc
    )

    return {
        "username": username,
        "display_name": (full_name if isinstance(full_name, str) and full_name else
                         meta.get("og:title", "").replace(f"(@{username})", "").strip(" -•")),
        "bio": _clean_text(bio_text, 1500),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": followers,
        "following_count": following,
        "post_count": posts,
        "is_private": is_private,
        "is_verified": is_verified,
        "account_type": "instagram_user",
        "extraction_method": "instagram_html_og_fallback",
    }


def extract_instagram(url: str) -> Dict[str, Any]:
    """Extraction Instagram avec stratégie multi-niveaux."""
    username = _instagram_username_from_url(url)
    if not username:
        raise ValueError("URL Instagram invalide (username manquant).")

    # 1) API publique web_profile_info
    data = _instagram_via_api(username)
    if data and (data.get("follower_count") or data.get("post_count") or data.get("bio")):
        return data

    # 2) Fallback HTML
    data_html = _instagram_via_html(url, username)
    if data_html:
        # Si l'API a retourné un username/avatar mais 0 partout, on enrichit avec le HTML
        if data and not (data.get("follower_count") or data.get("post_count")):
            for k in ("follower_count", "following_count", "post_count", "bio",
                      "is_private", "is_verified", "avatar_url", "display_name"):
                if not data.get(k) and data_html.get(k):
                    data[k] = data_html[k]
            data["extraction_method"] = "instagram_api+html_merge"
            return data
        return data_html

    if data:  # API a renvoyé quelque chose même si vide
        return data

    # 3) Last-resort
    return {
        "username": username,
        "display_name": username,
        "bio": "",
        "avatar_url": None,
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "instagram_user",
        "extraction_method": "instagram_username_only",
        "extraction_warning": (
            "Instagram a refusé l'accès non-authentifié. "
            "Seul le username a pu être déduit de l'URL."
        ),
    }


# ----------------------------------------------------------------------------- #
#  EXTRACTEUR TWITTER / X                                                        #
# ----------------------------------------------------------------------------- #
#
# Stratégie :
#   1) Endpoint syndication officiel : cdn.syndication.twimg.com/timeline/profile
#      -> renvoie un JSON public avec name, screen_name, description, statuses_count,
#         followers_count, friends_count, etc. (sans authentification).
#   2) Fallback Nitter (instances publiques en cascade).
#   3) Fallback OG (rarement utile, X bloque les bots sans JS).
# ----------------------------------------------------------------------------- #

def _twitter_username_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in (parsed.path or "").split("/") if p]
    if not parts:
        return ""
    p = parts[0].lower()
    # Évite les pages spéciales
    if p in ("home", "explore", "notifications", "messages", "i", "search",
             "settings", "compose", "intent", "share"):
        return ""
    return parts[0]


def _twitter_via_syndication(username: str) -> Optional[Dict[str, Any]]:
    """
    Endpoint Twitter syndication public (sans auth).
    On essaye d'abord https://syndication.twitter.com/srv/timeline-profile/screen-name/{user}
    qui renvoie du HTML avec un état React inliné (le plus riche), puis le CDN JSON.
    """
    if not username:
        return None

    # --- 1) syndication.twitter.com (HTML + JSON inliné) ---
    html_url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{quote(username)}"
    try:
        r = http_safe_get(html_url, referer=f"https://twitter.com/{username}")
    except Exception:
        r = None
    if r is not None and r.status_code == 200 and r.text:
        html = r.text
        # État inliné sous forme JSON dans __NEXT_DATA__
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
                      html, flags=re.DOTALL)
        if m:
            try:
                blob = json.loads(m.group(1))
                # Parcours récursif à la recherche d'un objet user
                def _walk(node):
                    if isinstance(node, dict):
                        if ("screen_name" in node and "followers_count" in node):
                            return node
                        for v in node.values():
                            r = _walk(v)
                            if r is not None:
                                return r
                    elif isinstance(node, list):
                        for x in node:
                            r = _walk(x)
                            if r is not None:
                                return r
                    return None
                user = _walk(blob)
                if user:
                    return _twitter_user_to_profile(user, username)
            except (ValueError, TypeError):
                pass
        # Fallback : regex sur le HTML brut
        followers = _safe_int((re.search(r'"followers_count"\s*:\s*(\d+)', html) or [0, 0])[1] if re.search(r'"followers_count"\s*:\s*(\d+)', html) else 0)
        friends   = _safe_int((re.search(r'"friends_count"\s*:\s*(\d+)', html) or [0, 0])[1] if re.search(r'"friends_count"\s*:\s*(\d+)', html) else 0)
        statuses  = _safe_int((re.search(r'"statuses_count"\s*:\s*(\d+)', html) or [0, 0])[1] if re.search(r'"statuses_count"\s*:\s*(\d+)', html) else 0)
        if followers or statuses:
            name_m = re.search(r'"name"\s*:\s*"([^"]+)"', html)
            bio_m  = re.search(r'"description"\s*:\s*"([^"]*)"', html)
            verified_m = re.search(r'"verified"\s*:\s*(true|false)', html)
            avatar_m = re.search(r'"profile_image_url_https"\s*:\s*"([^"]+)"', html)
            return {
                "username": username,
                "display_name": name_m.group(1) if name_m else username,
                "bio": _clean_text((bio_m.group(1) if bio_m else "").encode().decode('unicode_escape', errors='ignore'), 800),
                "avatar_url": (avatar_m.group(1).replace("_normal", "_400x400") if avatar_m else ""),
                "profile_url": f"https://x.com/{username}",
                "follower_count": followers,
                "following_count": friends,
                "post_count": statuses,
                "is_private": False,
                "is_verified": (verified_m.group(1) == "true") if verified_m else False,
                "account_type": "twitter_user",
                "extraction_method": "twitter_syndication_html",
            }

    # --- 2) cdn.syndication.twimg.com (JSON pur) ---
    api_url = (
        "https://cdn.syndication.twimg.com/timeline/profile"
        f"?screen_name={quote(username)}&dnt=true&suppress_response_codes=true"
    )
    headers = {"Origin": "https://twitter.com"}
    payload = http_safe_json(api_url, headers=headers,
                             referer=f"https://twitter.com/{username}")
    if not payload or not isinstance(payload, dict):
        return None

    # Différentes formes possibles selon les mises à jour de l'endpoint
    user = (payload.get("user")
            or (payload.get("headers") or {}).get("user")
            or {})
    # parfois la racine elle-même est l'objet user
    if not user and "screen_name" in payload:
        user = payload
    if not user:
        # parfois l'info est dans la première tweet
        tweets = payload.get("body") or payload.get("tweets") or []
        if isinstance(tweets, list) and tweets:
            user = (tweets[0] or {}).get("user") or {}
    if not user:
        return None
    return _twitter_user_to_profile(user, username)


def _twitter_user_to_profile(user: Dict[str, Any], username: str) -> Dict[str, Any]:
    """Normalise un objet 'user' Twitter (depuis syndication/Nitter) en profil interne."""
    followers = _safe_int(user.get("followers_count"))
    following = _safe_int(user.get("friends_count"))
    statuses  = _safe_int(user.get("statuses_count"))
    favourites= _safe_int(user.get("favourites_count"))
    avatar    = (user.get("profile_image_url_https")
                 or user.get("profile_image_url") or "")
    if avatar and "_normal" in avatar:
        avatar = avatar.replace("_normal", "_400x400")
    created   = user.get("created_at") or ""
    return {
        "username":        user.get("screen_name") or username,
        "display_name":    user.get("name") or user.get("screen_name") or username,
        "bio":             _clean_text(user.get("description"), 800),
        "avatar_url":      avatar,
        "profile_url":     f"https://x.com/{user.get('screen_name', username)}",
        "follower_count":  followers,
        "following_count": following,
        "post_count":      statuses,
        "favourites_count": favourites,
        "is_private":      bool(user.get("protected", False)),
        "is_verified":     bool(user.get("verified", False) or user.get("is_blue_verified", False)),
        "location":        user.get("location") or "",
        "url":             user.get("url") or "",
        "created_at":      created,
        "account_type":    "twitter_user",
        "extraction_method": "twitter_syndication_api",
    }


def _twitter_via_nitter(username: str) -> Optional[Dict[str, Any]]:
    """Fallback via une instance Nitter publique."""
    if not username:
        return None
    for base in NITTER_INSTANCES:
        url = f"{base}/{username}"
        try:
            r = http_safe_get(url)
        except Exception:
            continue
        if r is None or r.status_code >= 400:
            continue
        soup = BeautifulSoup(r.text, "html.parser")

        # Nitter expose la fullname et la bio dans des classes connues
        name_node = soup.select_one(".profile-card-fullname") or soup.select_one(".fullname")
        bio_node  = soup.select_one(".profile-bio")
        avatar    = soup.select_one(".profile-card-avatar img") or soup.select_one(".avatar")
        location  = soup.select_one(".profile-location")
        join_date = soup.select_one(".profile-joindate")

        # Compteurs
        def _stat(name: str) -> int:
            node = soup.select_one(f".profile-statlist .{name} .profile-stat-num")
            if not node:
                # variante : la stat list est une liste avec .profile-stat-header
                for li in soup.select(".profile-statlist li"):
                    head = (li.select_one(".profile-stat-header") or li).get_text(strip=True).lower()
                    if name.replace("-", "").replace("_", "") in head.replace(" ", "").replace("-", "").lower():
                        num = li.select_one(".profile-stat-num")
                        if num:
                            return _safe_int(num.get_text(strip=True))
            return _safe_int(node.get_text(strip=True)) if node else 0

        followers = _stat("followers")
        following = _stat("following")
        statuses  = _stat("posts") or _stat("tweets")

        if not (name_node or bio_node or followers or statuses):
            continue  # cette instance n'a rien rendu

        avatar_url = ""
        if avatar and avatar.get("src"):
            src = avatar["src"]
            avatar_url = src if src.startswith("http") else (base + src)

        return {
            "username": username,
            "display_name": (name_node.get_text(strip=True) if name_node else username),
            "bio": _clean_text(bio_node.get_text(" ", strip=True) if bio_node else "", 800),
            "avatar_url": avatar_url,
            "profile_url": f"https://x.com/{username}",
            "follower_count": followers,
            "following_count": following,
            "post_count": statuses,
            "is_private": False,
            "is_verified": bool(soup.select_one(".verified-icon")),
            "location": location.get_text(strip=True) if location else "",
            "created_at": join_date.get("title") if join_date and join_date.get("title") else
                          (join_date.get_text(strip=True) if join_date else ""),
            "account_type": "twitter_user",
            "extraction_method": f"twitter_nitter ({urlparse(base).netloc})",
        }
    return None


def extract_twitter(url: str) -> Dict[str, Any]:
    """Extraction Twitter / X (syndication > Nitter > OG)."""
    username = _twitter_username_from_url(url)
    if not username:
        raise ValueError("URL Twitter/X invalide (username manquant).")

    # 1) Syndication officielle
    data = _twitter_via_syndication(username)
    if data and (data.get("follower_count") or data.get("post_count") or data.get("bio")):
        return data

    # 2) Nitter
    data_nitter = _twitter_via_nitter(username)
    if data_nitter:
        if data and not (data.get("follower_count") or data.get("post_count")):
            for k in ("follower_count", "following_count", "post_count", "bio",
                      "is_verified", "avatar_url", "display_name", "location"):
                if not data.get(k) and data_nitter.get(k):
                    data[k] = data_nitter[k]
            data["extraction_method"] = "twitter_syndication+nitter_merge"
            return data
        return data_nitter

    # 3) OG en dernier recours
    r = http_safe_get(url)
    if r is not None and r.status_code < 400:
        soup = BeautifulSoup(r.text, "html.parser")
        meta = _extract_og_meta(soup)
        desc = meta.get("og:description") or meta.get("description") or ""
        return {
            "username": username,
            "display_name": meta.get("og:title", "").replace(f"(@{username})", "").strip(" -•") or username,
            "bio": _clean_text(desc, 800),
            "avatar_url": meta.get("og:image"),
            "profile_url": url,
            "follower_count": 0,
            "following_count": 0,
            "post_count": 0,
            "is_private": False,
            "is_verified": False,
            "account_type": "twitter_user",
            "extraction_method": "twitter_og_fallback",
            "extraction_warning": "X.com a refusé l'accès non-authentifié. Bio/compteurs probablement incomplets.",
        }

    if data:
        return data

    return {
        "username": username,
        "display_name": username,
        "bio": "",
        "avatar_url": None,
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "twitter_user",
        "extraction_method": "twitter_username_only",
        "extraction_warning": "Aucune source publique (syndication/Nitter/OG) n'a répondu.",
    }


# ----------------------------------------------------------------------------- #
#  EXTRACTEUR GITHUB                                                             #
# ----------------------------------------------------------------------------- #
#
# Stratégie :
#   1) API officielle GitHub (la plus fiable et stable).
#   2) Enrichissement avec /users/{login}/repos pour les langages.
#   3) Compteur commits réel via Search API (best-effort).
#   4) Liste des organisations publiques.
#   5) Fallback HTML si l'API est indisponible.
# ----------------------------------------------------------------------------- #

def _github_api_headers() -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENTS[0],
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _github_real_commit_count(username: str) -> int:
    """
    Compteur commits "best-effort" via Search API.
    Note : GitHub Search renvoie un total_count parfois aberrant (recherche
    floue côté backend) -> on borne la valeur à 10 millions et on cross-check
    avec le login exact du premier hit.
    """
    api = f"https://api.github.com/search/commits?q=author:{quote(username)}&per_page=1"
    try:
        resp = requests.get(api, headers=_github_api_headers(), timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json() or {}
            total = _safe_int(data.get("total_count"))
            # Vérification : le 1er commit retourné doit bien être de l'utilisateur
            items = data.get("items") or []
            if items:
                first_author = ((items[0] or {}).get("author") or {}).get("login", "")
                if first_author and first_author.lower() != username.lower():
                    # Auteur incorrect -> la recherche est floue, on ignore
                    return 0
            # Borne haute raisonnable (1M commits = ~30 par jour pendant 90 ans)
            return min(total, 1_000_000)
    except requests.RequestException as exc:
        print(f"[GitHub commit-search] indisponible : {exc}", flush=True)
    return 0


def _github_orgs(username: str) -> List[str]:
    api = f"https://api.github.com/users/{quote(username)}/orgs"
    try:
        resp = requests.get(api, headers=_github_api_headers(), timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            return [str(o.get("login", "")) for o in (resp.json() or []) if isinstance(o, dict)]
    except requests.RequestException:
        pass
    return []


def _github_top_languages(username: str) -> List[str]:
    """Langages distincts utilisés dans les dépôts publics (max 30 repos analysés)."""
    api = f"https://api.github.com/users/{quote(username)}/repos?per_page=30&sort=updated"
    try:
        resp = requests.get(api, headers=_github_api_headers(), timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            langs = []
            for repo in (resp.json() or []):
                if isinstance(repo, dict):
                    lang = repo.get("language")
                    if lang and lang not in langs:
                        langs.append(lang)
            return langs
    except requests.RequestException:
        pass
    return []


def extract_github(url: str) -> Dict[str, Any]:
    """Extraction GitHub via API officielle (avec enrichissement)."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise ValueError("URL GitHub invalide (login manquant).")
    username = parts[0]

    api_url = f"https://api.github.com/users/{quote(username)}"
    try:
        api_resp = requests.get(api_url, headers=_github_api_headers(), timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        api_resp = None
        print(f"[GitHub API] erreur réseau : {exc}", flush=True)

    if api_resp is not None and api_resp.status_code == 200:
        data = api_resp.json()

        # Compteur commits via /events (90 derniers jours, max 100 events)
        commits_estimate = 0
        try:
            ev = requests.get(
                f"https://api.github.com/users/{quote(username)}/events/public?per_page=100",
                headers=_github_api_headers(), timeout=HTTP_TIMEOUT,
            )
            if ev.status_code == 200:
                events = ev.json() or []
                commits_estimate = sum(
                    len((e.get("payload") or {}).get("commits") or [])
                    for e in events if isinstance(e, dict) and e.get("type") == "PushEvent"
                )
        except requests.RequestException:
            pass

        # Compteur commits "réel" via Search API
        commits_search = _github_real_commit_count(username)
        if commits_search > commits_estimate:
            num_commit = commits_search
        else:
            num_commit = commits_estimate

        orgs = _github_orgs(username)
        langs = _github_top_languages(username)

        # active_days approximatif : (updated_at - created_at) en jours
        active_days = 0
        try:
            from datetime import datetime as _dt
            if data.get("created_at") and data.get("updated_at"):
                c = _dt.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                u = _dt.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
                active_days = max(0, (u - c).days)
        except Exception:
            pass

        return {
            "username":        data.get("login", username),
            "display_name":    data.get("name") or data.get("login"),
            "bio":             _clean_text(data.get("bio"), 800),
            "avatar_url":      data.get("avatar_url"),
            "profile_url":     data.get("html_url", url),
            "follower_count":  _safe_int(data.get("followers")),
            "following_count": _safe_int(data.get("following")),
            "post_count":      _safe_int(data.get("public_repos")),
            "public_repos":    _safe_int(data.get("public_repos")),
            "public_gists":    _safe_int(data.get("public_gists")),
            "email":           data.get("email") or "",
            "company":         data.get("company") or "",
            "blog":            data.get("blog") or "",
            "twitter_username": data.get("twitter_username") or "",
            "location":        data.get("location") or "",
            "hireable":        bool(data.get("hireable", False)),
            "created_at":      data.get("created_at"),
            "updated_at":      data.get("updated_at"),
            "active_days":     active_days,
            "num_commit_estimate": commits_estimate,
            "num_commit":      num_commit,
            "num_activities":  commits_estimate + _safe_int(data.get("public_repos")),
            "organizations":   orgs,
            "languages":       langs,
            "is_private":      False,
            "is_verified":     bool(data.get("site_admin", False)),
            "account_type":    "github_org" if data.get("type") == "Organization" else "github_user",
            "extraction_method": "github_official_api",
        }

    if api_resp is not None and api_resp.status_code == 404:
        raise ValueError(f"Compte GitHub introuvable : {username}")

    # Fallback HTML si l'API échoue (rate-limit par ex.)
    r = http_safe_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL GitHub.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    name_node = soup.select_one(".p-name") or soup.select_one("[itemprop='name']")
    bio_node  = soup.select_one(".p-note") or soup.select_one("div.user-profile-bio")

    # Compteurs visibles dans le HTML
    def _stat_html(label: str) -> int:
        node = soup.find("a", href=re.compile(rf"\?tab={label}$"))
        if not node:
            return 0
        num = node.select_one(".Counter")
        return _safe_int(num.get("title") or num.get_text(strip=True)) if num else 0

    followers = _stat_html("followers")
    following = _stat_html("following")
    repos     = _stat_html("repositories")

    return {
        "username": username,
        "display_name": ((name_node.get_text(strip=True) if name_node else "")
                         or meta.get("og:title", "").split("·")[0].strip()),
        "bio": _clean_text((bio_node.get_text(strip=True) if bio_node else
                            meta.get("description")), 600),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": followers,
        "following_count": following,
        "post_count": repos,
        "public_repos": repos,
        "num_commit_estimate": 0,
        "num_commit": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "github_user",
        "extraction_method": "github_html_fallback",
        "extraction_warning": "API GitHub indisponible (rate-limit ?). Données extraites du HTML.",
    }


# ----------------------------------------------------------------------------- #
#  EXTRACTEUR LINKEDIN                                                           #
# ----------------------------------------------------------------------------- #
#
# LinkedIn bloque très agressivement les visites non-authentifiées. On combine :
#   1) oEmbed officiel (https://www.linkedin.com/oembed?...) : renvoie un JSON
#      avec title + html (renvoie souvent un thumbnail + iframe embed).
#   2) Récupération HTML + extraction OG/Twitter + microdata (itemprop="*") +
#      JSON-LD + scan JSON inliné.
#   3) Cas particuliers : pages /in/, /company/, /school/, /pub/.
# ----------------------------------------------------------------------------- #

def _linkedin_slug_from_url(url: str) -> Tuple[str, str]:
    """Renvoie (slug, type) où type ∈ {in, company, school, pub, generic}."""
    parsed = urlparse(url)
    parts = [p for p in (parsed.path or "").split("/") if p]
    if not parts:
        return "", "generic"
    container = parts[0].lower()
    if container in ("in", "company", "school", "pub"):
        slug = parts[1] if len(parts) >= 2 else ""
        return slug, container
    return parts[-1], "generic"


def _linkedin_via_oembed(url: str) -> Optional[Dict[str, Any]]:
    """oEmbed officiel (renvoie un JSON limité mais propre)."""
    api_url = f"https://www.linkedin.com/oembed?url={quote(url, safe='')}&format=json"
    payload = http_safe_json(api_url, referer="https://www.linkedin.com/")
    if not payload or not isinstance(payload, dict):
        return None
    title = payload.get("title") or ""
    html_blob = payload.get("html") or ""
    # Le 'html' contient parfois la fonction + une description courte
    soup_oe = BeautifulSoup(html_blob, "html.parser") if html_blob else None
    bio_oe = ""
    if soup_oe is not None:
        for tag in soup_oe(["script", "style"]):
            tag.decompose()
        bio_oe = soup_oe.get_text(" ", strip=True)
    return {
        "_oembed_title": title.strip(),
        "_oembed_bio":   _clean_text(bio_oe, 800),
        "_oembed_author_name": payload.get("author_name") or "",
        "_oembed_author_url":  payload.get("author_url") or "",
        "_oembed_thumbnail":   payload.get("thumbnail_url") or payload.get("image") or "",
    }


def extract_linkedin(url: str) -> Dict[str, Any]:
    """Extraction LinkedIn via oEmbed + OG + Microdata + JSON-LD."""
    slug, kind = _linkedin_slug_from_url(url)

    # 1) oEmbed (best-effort)
    oe = _linkedin_via_oembed(url) or {}

    # 2) Page HTML
    r = http_safe_get(url, referer="https://www.linkedin.com/")
    html_data = {
        "username": slug,
        "display_name": oe.get("_oembed_title") or slug,
        "bio": oe.get("_oembed_bio") or "",
        "avatar_url": oe.get("_oembed_thumbnail") or None,
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "headline": "",
        "company": "",
        "industry": "",
        "location": "",
        "skills": [],
        "experiences": [],
        "educations": [],
        "account_type": ("linkedin_profile" if kind == "in"
                         else ("linkedin_company" if kind == "company"
                               else ("linkedin_school" if kind == "school"
                                     else "linkedin_other"))),
        "extraction_method": "linkedin_oembed_only" if oe else "linkedin_username_only",
    }

    if r is None or r.status_code >= 400:
        if not (oe.get("_oembed_title") or oe.get("_oembed_bio")):
            html_data["extraction_warning"] = (
                "LinkedIn a refusé l'accès non-authentifié et oEmbed n'a rien renvoyé."
            )
        return html_data

    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    jsonld = _extract_jsonld(soup)
    microdata = _extract_microdata(soup)
    raw_html = r.text

    # -- OG / Twitter card --
    if meta.get("og:title"):
        html_data["display_name"] = meta["og:title"].split("|")[0].strip()
    if meta.get("og:image"):
        html_data["avatar_url"] = meta["og:image"]
    main_bio = meta.get("og:description") or meta.get("description") or ""

    # -- JSON-LD (Person / Organization / EducationalOrganization) --
    followers = 0
    bio_parts: List[str] = [main_bio]
    if html_data["bio"]:
        bio_parts.append(html_data["bio"])
    skills_set: List[str] = []
    experiences: List[str] = []
    educations: List[str] = []

    for item in jsonld:
        itype = item.get("@type") or ""
        if isinstance(itype, list):
            itype = " ".join(itype)
        # Nom officiel
        if item.get("name") and not html_data.get("display_name"):
            html_data["display_name"] = str(item["name"]).strip()
        # Image
        img = item.get("image")
        if isinstance(img, dict):
            img = img.get("url") or img.get("contentUrl")
        if isinstance(img, str) and not html_data.get("avatar_url"):
            html_data["avatar_url"] = img

        # Followers / interactionCount
        for key in ("followerCount", "interactionCount", "userInteractionCount"):
            if key in item:
                followers = max(followers, _safe_int(item.get(key)))

        # Texte
        for key in ("description", "jobTitle", "headline", "alumniOf", "worksFor",
                    "knowsAbout", "skills"):
            v = item.get(key)
            if isinstance(v, str):
                bio_parts.append(v)
                if key == "headline" and not html_data["headline"]:
                    html_data["headline"] = v
                if key == "jobTitle" and not html_data["headline"]:
                    html_data["headline"] = v
                if key == "knowsAbout":
                    skills_set.append(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str):
                        bio_parts.append(x)
                        if key == "knowsAbout":
                            skills_set.append(x)
                    elif isinstance(x, dict):
                        name = x.get("name") or x.get("title")
                        if name:
                            bio_parts.append(str(name))
                            if key == "knowsAbout":
                                skills_set.append(str(name))
                            if key == "alumniOf":
                                educations.append(str(name))
                            if key == "worksFor":
                                experiences.append(str(name))
            elif isinstance(v, dict):
                name = v.get("name")
                if name:
                    bio_parts.append(str(name))
                    if key == "worksFor" and not html_data["company"]:
                        html_data["company"] = str(name)

        # Localisation
        addr = item.get("address") or item.get("location")
        if isinstance(addr, dict):
            loc = (addr.get("addressLocality") or addr.get("name") or "")
            if loc and not html_data["location"]:
                html_data["location"] = str(loc)
        elif isinstance(addr, str) and not html_data["location"]:
            html_data["location"] = addr

        # Industrie
        if item.get("industry") and not html_data["industry"]:
            html_data["industry"] = str(item["industry"])

    # -- Microdata HTML --
    if microdata:
        if not html_data["display_name"] and microdata.get("name"):
            v = microdata["name"]
            html_data["display_name"] = (v if isinstance(v, str) else v[0])
        if not html_data["headline"] and microdata.get("jobTitle"):
            v = microdata["jobTitle"]
            html_data["headline"] = v if isinstance(v, str) else v[0]
        if not html_data["company"] and microdata.get("worksFor"):
            v = microdata["worksFor"]
            html_data["company"] = v if isinstance(v, str) else v[0]
        if microdata.get("description"):
            v = microdata["description"]
            bio_parts.append(v if isinstance(v, str) else " ".join(v))

    # -- Scan JSON inliné (LinkedIn inline beaucoup d'état dans un JSON volumineux) --
    if followers == 0:
        v = _find_in_html_json(raw_html, "followerCount", "numFollowers", "followingCount")
        if v is not None:
            followers = _safe_int(v)
    headline_inline = _find_in_html_json(raw_html, "headline", "occupation")
    if isinstance(headline_inline, str) and not html_data["headline"]:
        html_data["headline"] = headline_inline

    # -- Construction finale --
    html_data["follower_count"] = followers
    html_data["bio"] = _clean_text(" ".join(p for p in bio_parts if p), 4000)
    html_data["skills"] = list(dict.fromkeys(skills_set))[:30]
    html_data["experiences"] = list(dict.fromkeys(experiences))[:15]
    html_data["educations"] = list(dict.fromkeys(educations))[:10]

    # Détection d'une page de login (LinkedIn redirige souvent)
    if (("authwall" in raw_html.lower() or "sign-in" in raw_html.lower()[:5000])
            and not html_data["bio"] and not html_data["headline"]):
        html_data["extraction_warning"] = (
            "LinkedIn a renvoyé une page de connexion. Données limitées au "
            "thumbnail oEmbed et au titre OG."
        )
        html_data["extraction_method"] = "linkedin_oembed+authwall"
    else:
        html_data["extraction_method"] = "linkedin_oembed+html_og+jsonld+microdata"

    return html_data


# ----------------------------------------------------------------------------- #
#  EXTRACTEUR OFFRES D'EMPLOI                                                    #
# ----------------------------------------------------------------------------- #

def _extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.IGNORECASE)
    return m.group(0) if m else None


def extract_job_posting(url: str) -> Dict[str, Any]:
    """
    Extraction d'une offre d'emploi via JSON-LD JobPosting + OpenGraph
    + microdonnées HTML (itemprop) + texte brut.
    """
    r = http_safe_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL de l'offre d'emploi.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    jsonld = _extract_jsonld(soup)
    microdata = _extract_microdata(soup)

    job_payload = next(
        (j for j in jsonld if (j.get("@type") == "JobPosting"
                               or (isinstance(j.get("@type"), list)
                                   and "JobPosting" in j["@type"]))),
        {},
    )

    # Texte brut (utilisé par le LSTM)
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    page_text = _clean_text(soup.get_text(" ", strip=True), limit=12000)

    # Description : JSON-LD > microdata > OG > texte brut
    description = (job_payload.get("description")
                   or microdata.get("description")
                   or meta.get("og:description") or "")
    if isinstance(description, list):
        description = " ".join(str(x) for x in description)
    description = _clean_text(re.sub(r"<[^>]+>", " ", str(description)), limit=12000)
    if not description or len(description) < 100:
        # On replie sur le texte brut de la page
        description = page_text

    # Titre
    title = (job_payload.get("title")
             or microdata.get("title")
             or meta.get("og:title") or meta.get("title") or "")
    if isinstance(title, list):
        title = title[0]

    # Hiring organization
    hiring_org = job_payload.get("hiringOrganization") or {}
    if isinstance(hiring_org, dict):
        hiring_name = hiring_org.get("name")
    else:
        hiring_name = str(hiring_org) if hiring_org else None
    if not hiring_name:
        hiring_name = (microdata.get("hiringOrganization")
                       or microdata.get("organization")
                       or microdata.get("name"))
        if isinstance(hiring_name, list):
            hiring_name = hiring_name[0]

    # Job location
    job_location = job_payload.get("jobLocation") or {}
    job_location_str = None
    if isinstance(job_location, dict):
        addr = job_location.get("address") or {}
        if isinstance(addr, dict):
            job_location_str = ", ".join(
                str(addr.get(k)) for k in
                ("addressLocality", "addressRegion", "addressCountry")
                if addr.get(k)
            )
        else:
            job_location_str = str(job_location)
    elif isinstance(job_location, list):
        job_location_str = "; ".join(str(x) for x in job_location)
    elif job_location:
        job_location_str = str(job_location)
    if not job_location_str:
        v = microdata.get("jobLocation") or microdata.get("addressLocality")
        if v:
            job_location_str = v if isinstance(v, str) else " / ".join(v)

    # Salary
    salary_str = None
    salary = job_payload.get("baseSalary")
    if isinstance(salary, dict):
        currency = salary.get("currency", "")
        val = salary.get("value") or {}
        if isinstance(val, dict):
            salary_str = (f"{val.get('minValue', '')}-{val.get('maxValue', '')} "
                          f"{currency}").strip()
        else:
            salary_str = f"{val} {currency}".strip()
    elif salary:
        salary_str = str(salary)
    if not salary_str:
        v = microdata.get("baseSalary") or microdata.get("salary")
        if v:
            salary_str = v if isinstance(v, str) else " ".join(str(x) for x in v)

    # Date posted
    date_posted = (job_payload.get("datePosted") or microdata.get("datePosted")
                   or job_payload.get("dateCreated"))
    if isinstance(date_posted, list):
        date_posted = date_posted[0]

    valid_through = (job_payload.get("validThrough") or microdata.get("validThrough"))
    if isinstance(valid_through, list):
        valid_through = valid_through[0]

    employment_type = (job_payload.get("employmentType") or microdata.get("employmentType"))
    if isinstance(employment_type, list):
        employment_type = ", ".join(str(x) for x in employment_type)

    # Email de contact
    contact_email = _extract_email(description) or _extract_email(page_text)

    # Texte final passé au LSTM (description si suffisante, sinon page complète)
    job_text_final = description if len(description) > 200 else page_text

    return {
        "is_job_posting": True,
        "title":               _clean_text(str(title), 300),
        "description":         _clean_text(description, 6000),
        "hiring_organization": _clean_text(str(hiring_name) if hiring_name else "", 300),
        "job_location":        _clean_text(str(job_location_str) if job_location_str else "", 300),
        "date_posted":         date_posted,
        "valid_through":       valid_through,
        "employment_type":     employment_type,
        "base_salary":         _clean_text(str(salary_str) if salary_str else "", 200),
        "contact_email":       contact_email,
        "apply_url":           job_payload.get("url") or url,
        "profile_url":         url,
        "job_text":            job_text_final,
        "extraction_method": ("job_jsonld" if job_payload else
                              ("job_microdata" if microdata else "job_html_text")),
    }


# ----------------------------------------------------------------------------- #
#  EXTRACTEUR GÉNÉRIQUE                                                          #
# ----------------------------------------------------------------------------- #

def extract_generic(url: str) -> Dict[str, Any]:
    """Extraction générique (fallback) sur n'importe quelle page web."""
    r = http_safe_get(url)
    if r is None:
        raise ValueError("Impossible d'accéder à l'URL fournie.")
    soup = BeautifulSoup(r.text, "html.parser")
    meta = _extract_og_meta(soup)
    title = (soup.title.string if soup.title else "") or meta.get("og:title", "")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    page_text = _clean_text(soup.get_text(" ", strip=True), limit=6000)
    return {
        "username": urlparse(url).netloc,
        "display_name": _clean_text(title, 200),
        "bio": _clean_text(meta.get("og:description") or meta.get("description"), 800),
        "avatar_url": meta.get("og:image"),
        "profile_url": url,
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
        "is_private": False,
        "is_verified": False,
        "account_type": "generic_web",
        "page_text_excerpt": page_text[:800],
        "extraction_method": "generic_html_og",
    }


EXTRACTORS = {
    "instagram": extract_instagram,
    "linkedin":  extract_linkedin,
    "twitter":   extract_twitter,
    "github":    extract_github,
    "job":       extract_job_posting,
    "generic":   extract_generic,
}
# ----------------------------------------------------------------------------- #
#  CONSTRUCTION DES FEATURES POUR LES MODÈLES RANDOM FOREST                      #
# ----------------------------------------------------------------------------- #

# --- Instagram (et Twitter) : 8 features numériques dans l'ordre du scaler ------
INSTAGRAM_FEATURE_ORDER = [
    "userFollowerCount",
    "userFollowingCount",
    "userBiographyLength",
    "userMediaCount",
    "userHasProfilPic",
    "userIsPrivate",
    "usernameDigitCount",
    "usernameLength",
]


def _instagram_features(f: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Construit le vecteur (1,8) Instagram-style et renvoie aussi les valeurs brutes."""
    vals = {
        "userFollowerCount":   float(_safe_int(f.get("userFollowerCount",   f.get("follower_count",  f.get("followers"))))),
        "userFollowingCount":  float(_safe_int(f.get("userFollowingCount",  f.get("following_count", f.get("following"))))),
        "userBiographyLength": float(_safe_int(f.get("userBiographyLength", f.get("bio_length",      0)))),
        "userMediaCount":      float(_safe_int(f.get("userMediaCount",      f.get("post_count",      0)))),
        "userHasProfilPic":    1.0 if _safe_bool(f.get("userHasProfilPic", f.get("has_profile_pic", True))) else 0.0,
        "userIsPrivate":       1.0 if _safe_bool(f.get("userIsPrivate",    f.get("is_private",      False))) else 0.0,
        "usernameDigitCount":  float(_safe_int(f.get("usernameDigitCount", f.get("username_digits", 0)))),
        "usernameLength":      float(_safe_int(f.get("usernameLength",     f.get("username_length", 0)))),
    }
    row = np.array([[vals[k] for k in INSTAGRAM_FEATURE_ORDER]], dtype=np.float32)
    return row, vals


def _linkedin_text_blob(f: Dict[str, Any]) -> str:
    """
    Concatène en un texte unique tous les champs textuels utiles d'un profil
    LinkedIn (bio, titre, compétences, expériences, etc.) pour passage TF-IDF.
    """
    parts: List[str] = []
    for key in (
        "bio", "headline", "summary", "about",
        "display_name", "name", "title",
        "skills", "experiences", "educations", "interests",
        "company", "industry", "location",
    ):
        v = f.get(key)
        if v is None:
            continue
        if isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        else:
            parts.append(str(v))
    blob = " ".join(parts).strip()
    return blob or (f.get("display_name") or f.get("username") or "")


def _bothawk_text_blob(f: Dict[str, Any]) -> str:
    """
    Construit la 'chaîne de tags' qui sera vectorisée par tfidf_bothawk.
    Le vocabulaire connu est :
      account, active, activity, bio, commit, content, days, email,
      followers, following, high, info, login, low, many, name,
      periodic, present, profile, repos, similar, single, tag
    On émet chaque token avec une intensité proportionnelle aux signaux observés
    sur le profil (heuristique inspirée de l'article Bothawk).
    """
    tokens: List[str] = []

    def emit(token: str, times: int = 1) -> None:
        tokens.extend([token] * max(0, int(times)))

    followers = _safe_int(f.get("follower_count", f.get("Number of followers", 0)))
    following = _safe_int(f.get("following_count", f.get("Number of following", 0)))
    repos     = _safe_int(f.get("num_repository",  f.get("public_repos", 0)))
    commits   = _safe_int(f.get("num_commit",      f.get("num_commit_estimate", 0)))
    active_d  = _safe_int(f.get("active_days", 0))
    activities= _safe_int(f.get("num_activities", commits + repos))

    bio       = str(f.get("bio") or "").strip()
    email     = str(f.get("email") or "").strip()
    login     = str(f.get("login") or f.get("username") or "").strip()
    name      = str(f.get("name") or f.get("display_name") or "").strip()
    has_pic   = _safe_bool(f.get("has_profile_pic", bool(f.get("avatar_url"))))

    # Présence de champs (info / present)
    emit("info",    1)
    emit("profile", 1)
    if name:  emit("name",    1); emit("present", 1)
    if login: emit("login",   1); emit("present", 1)
    if bio:   emit("bio",     1); emit("content", 1)
    if email: emit("email",   1); emit("present", 1)
    if has_pic: emit("present", 1)

    # Activité
    if commits   > 0: emit("commit",   1 + min(commits // 50, 4))
    if repos     > 0: emit("repos",    1 + min(repos // 5, 4))
    if activities> 0: emit("activity", 1 + min(activities // 30, 4))
    if active_d  > 0: emit("active",   1 + min(active_d // 30, 4)); emit("days", 1)

    # Volume followers/following
    if followers > 0: emit("followers", 1 + min(followers // 100, 4))
    if following > 0: emit("following", 1 + min(following // 100, 4))

    # Signaux qualitatifs
    if followers > 5000:           emit("high", 2)
    elif followers < 5:            emit("low", 2); emit("single", 1)
    if following > 0 and followers > 0:
        ratio = following / max(followers, 1)
        if ratio > 10:             emit("many", 2)
        elif ratio < 0.05:         emit("similar", 2)
    if active_d > 0 and activities > 0:
        cadence = activities / max(active_d, 1)
        if cadence > 5:            emit("periodic", 2)
    if not bio and not email and not name:
        emit("single", 1); emit("tag", 1)
    # Mot 'account' toujours présent
    emit("account", 1)

    return " ".join(tokens)


# ----------------------------------------------------------------------------- #
#  CALCUL DU "RISK SCORE" + "SYNTHÈSE DES FACTEURS"                              #
# ----------------------------------------------------------------------------- #

# Seuils calibrés v7.2 : on est plus prudent pour réduire les faux positifs.
#   prob >= FAKE_THRESHOLD            -> "fake"
#   SUSPICIOUS_THRESHOLD <= prob < FAKE_THRESHOLD -> "suspicious"
#   prob < SUSPICIOUS_THRESHOLD       -> "genuine"
FAKE_THRESHOLD       = float(os.environ.get("FAKE_THRESHOLD", "0.65"))
SUSPICIOUS_THRESHOLD = float(os.environ.get("SUSPICIOUS_THRESHOLD", "0.45"))


def _normalize_risk(prob_fake: float) -> Tuple[int, str, float]:
    """Convertit une probabilité 'fake' en (score 0..100, label, confiance).

    v7.2 : 3 classes pour réduire les faux positifs :
      - fake       (prob >= 0.65)
      - suspicious (0.45 <= prob < 0.65)
      - genuine    (prob <  0.45)
    """
    prob_fake = max(0.0, min(1.0, float(prob_fake)))
    score = int(round(prob_fake * 100))
    if prob_fake >= FAKE_THRESHOLD:
        label = "fake"
        confidence = prob_fake
    elif prob_fake >= SUSPICIOUS_THRESHOLD:
        label = "suspicious"
        # Confiance plus faible dans la zone grise
        confidence = 0.5 + abs(prob_fake - 0.55) * 0.6
    else:
        label = "genuine"
        confidence = 1.0 - prob_fake
    return score, label, round(min(1.0, confidence), 3)


# ----------------------------------------------------------------------------- #
#  v7.2 : DATA-QUALITY GATE et SCORE HEURISTIQUE                                 #
# ----------------------------------------------------------------------------- #
#
# Idée : avant d'appeler le ML, on évalue la fiabilité des données extraites.
# Si l'extraction a clairement échoué (tout à 0, pas de bio, pas d'avatar...),
# on REFUSE de prédire et on renvoie 'insufficient_data'. Cela évite que des
# zéros soient interprétés à tort comme un profil bot.
#
# Par ailleurs, lorsqu'on a assez de données, on combine la probabilité du
# modèle ML avec un score heuristique calibré sur des règles métier.
# ----------------------------------------------------------------------------- #

def _build_insufficient_data_result(platform: str, model_key: str,
                                    features: Dict[str, Any],
                                    reason: str) -> Dict[str, Any]:
    """Réponse standardisée quand l'extraction est trop pauvre pour décider."""
    return {
        "model":             model_key,
        "platform":          platform,
        "prediction_score":  None,
        "risk_score":        None,
        "classification":    "insufficient_data",
        "confidence":        0.0,
        "metrics":           MODEL_PERFORMANCE.get(model_key, {}),
        "features":          features if isinstance(features, dict) else {},
        "shap_values":       {},
        "interpretation": {
            "verdict_humain":   "Données insuffisantes pour décider",
            "synthese_facteurs": [],
            "signaux_alerte":   [reason],
            "data_quality_issue": reason,
            "recommendation":   (
                "L'extraction publique n'a pas pu récupérer assez d'informations "
                "pour ce profil (la plateforme bloque les requêtes non authentifiées "
                "ou le profil est inaccessible). Réessayez plus tard, fournissez les "
                "informations manuellement via /api/analyze, ou utilisez l'API "
                "officielle de la plateforme."
            ),
        },
    }


def _instagram_data_quality(vals: Dict[str, float]) -> Tuple[bool, str]:
    """Vérifie si l'extraction Instagram/Twitter est exploitable.

    Renvoie (is_ok, raison_si_ko).
    """
    follower = vals.get("userFollowerCount", 0)
    following = vals.get("userFollowingCount", 0)
    posts = vals.get("userMediaCount", 0)
    bio_len = vals.get("userBiographyLength", 0)
    has_pic = vals.get("userHasProfilPic", 0)

    # Aucun signal observable : extraction très probablement bloquée
    if (follower == 0 and following == 0 and posts == 0
            and bio_len == 0 and has_pic == 0):
        return False, (
            "Tous les compteurs sont à 0 et aucune photo/bio détectée : "
            "l'extraction publique a été bloquée par la plateforme."
        )
    # Quasi rien : juste un avatar OG mais pas de chiffres
    if follower == 0 and posts == 0 and following == 0 and bio_len == 0:
        return False, (
            "Extraction partielle : aucun compteur n'a pu être récupéré "
            "(plateforme bloquée ou profil privé non indexé)."
        )
    return True, ""


def _instagram_heuristic_prob(vals: Dict[str, float]) -> float:
    """Score heuristique 0..1 calibré sur des règles métier observables.

    Indépendant du modèle ML : sert de stabilisateur pour réduire les faux
    positifs / faux négatifs liés au biais du jeu d'entraînement.
    """
    follower   = vals.get("userFollowerCount", 0)
    following  = vals.get("userFollowingCount", 0)
    bio_len    = vals.get("userBiographyLength", 0)
    posts      = vals.get("userMediaCount", 0)
    has_pic    = vals.get("userHasProfilPic", 0)
    digits     = vals.get("usernameDigitCount", 0)
    uname_len  = vals.get("usernameLength", 0)

    score = 0.0

    # Signal #1 : pas du tout de followers + suit beaucoup -> très suspect
    if follower <= 1 and following > 100:
        score += 0.35
    elif follower < 10 and following > 500:
        score += 0.25

    # Signal #2 : ratio following/follower très élevé (typique des bots)
    ratio_fr = following / max(follower, 1)
    if ratio_fr > 50 and follower < 50:
        score += 0.20
    elif ratio_fr > 10 and follower < 100:
        score += 0.10

    # Signal #3 : aucune publication
    if posts == 0:
        score += 0.15
    elif posts < 3:
        score += 0.05

    # Signal #4 : pas de photo de profil
    if has_pic == 0:
        score += 0.15

    # Signal #5 : pseudo avec beaucoup de chiffres
    if uname_len > 0:
        digit_ratio = digits / uname_len
        if digit_ratio > 0.5:
            score += 0.15
        elif digits >= 4:
            score += 0.08

    # Signal #6 : bio absente alors que le compte n'est pas tout neuf (posts > 0)
    if bio_len == 0 and posts > 5:
        score += 0.05

    # SIGNAUX NÉGATIFS (réduisent la suspicion)
    if follower > 10_000:           score -= 0.20
    elif follower > 1_000:          score -= 0.10
    if posts > 50:                  score -= 0.10
    if posts > 200:                 score -= 0.05
    if bio_len > 40 and has_pic:    score -= 0.10
    if 0 < ratio_fr < 2 and follower > 200:
        score -= 0.10  # ratio sain

    return max(0.0, min(1.0, 0.5 + score))  # centré sur 0.5


def _linkedin_data_quality(blob: str) -> Tuple[bool, str]:
    """Vérifie qu'on a assez de texte pour le TF-IDF LinkedIn."""
    if not blob or not blob.strip():
        return False, "Aucun contenu textuel exploitable pour le profil LinkedIn."
    words = re.findall(r"\w+", blob.lower())
    if len(words) < 25:
        return False, (
            f"Profil LinkedIn trop peu rempli ({len(words)} mots extraits, "
            f"minimum 25 requis). L'authwall LinkedIn bloque probablement "
            f"l'extraction publique des champs détaillés."
        )
    return True, ""


def _linkedin_heuristic_prob(blob: str, features: Dict[str, Any]) -> float:
    """Score heuristique LinkedIn 0..1."""
    blob_l = blob.lower()
    words = re.findall(r"\w+", blob_l)
    n_words = len(words)
    if n_words == 0:
        return 0.5

    score = 0.5

    # Très peu de texte
    if n_words < 40:    score += 0.10
    if n_words < 25:    score += 0.15

    # Mots-clés d'arnaque
    scam_kw = ["crypto", "bitcoin", "forex", "binary option", "make money",
               "passive income", "investment opportunity", "telegram",
               "whatsapp only", "earn from home", "guaranteed return"]
    hits = sum(1 for k in scam_kw if k in blob_l)
    score += min(0.30, hits * 0.10)

    # Texte répétitif (génération auto)
    if n_words > 5:
        rep = 1.0 - (len(set(words)) / n_words)
        if rep > 0.75:  score += 0.15
        elif rep > 0.60: score += 0.05

    # Signaux positifs : structure pro
    pro_kw = ["experience", "manager", "engineer", "university", "team",
              "skills", "led", "developed", "managed", "designed",
              "projets", "projet", "développé", "équipe", "ingénieur",
              "directeur", "université", "diplôme"]
    pro_hits = sum(1 for k in pro_kw if k in blob_l)
    if pro_hits >= 3:   score -= 0.15
    if pro_hits >= 6:   score -= 0.10

    return max(0.0, min(1.0, score))


def _bothawk_data_quality(features: Dict[str, Any]) -> Tuple[bool, str]:
    """Vérifie qu'on a assez d'informations GitHub pour décider."""
    has_login = bool((features.get("login") or features.get("username") or "").strip())
    repos     = _safe_int(features.get("num_repository",
                                       features.get("public_repos")))
    followers = _safe_int(features.get("follower_count"))
    following = _safe_int(features.get("following_count"))
    name      = str(features.get("name") or features.get("display_name") or "").strip()
    bio       = str(features.get("bio") or "").strip()

    if not has_login and not name and not bio and repos == 0:
        return False, (
            "Profil GitHub non récupéré (API GitHub a refusé la requête "
            "ou l'utilisateur n'existe pas)."
        )
    # Compte tout neuf sans aucune activité : on signale plutôt qu'on prédise
    if repos == 0 and followers == 0 and following == 0 and not bio:
        return False, (
            "Compte GitHub vide : aucun dépôt, follower, ou bio. Décision "
            "impossible à porter automatiquement (peut être un nouveau compte "
            "légitime)."
        )
    return True, ""


def _bothawk_heuristic_prob(features: Dict[str, Any]) -> float:
    """Score heuristique GitHub 0..1, calibré pour réduire les faux positifs."""
    repos     = _safe_int(features.get("num_repository",
                                       features.get("public_repos")))
    commits   = _safe_int(features.get("num_commit",
                                       features.get("num_commit_estimate")))
    followers = _safe_int(features.get("follower_count"))
    following = _safe_int(features.get("following_count"))
    bio       = str(features.get("bio") or "").strip()
    email     = str(features.get("email") or "").strip()
    name      = str(features.get("name") or features.get("display_name") or "").strip()
    active_d  = _safe_int(features.get("active_days"))
    orgs      = features.get("organizations") or []
    langs     = features.get("languages") or []

    score = 0.5

    # Signaux négatifs (=> réduisent la suspicion) : dev actif
    if repos > 5:    score -= 0.10
    if repos > 20:   score -= 0.10
    if commits > 50: score -= 0.10
    if commits > 200: score -= 0.05
    if bio:          score -= 0.05
    if email:        score -= 0.05
    if name:         score -= 0.05
    if followers > 10: score -= 0.05
    if followers > 100: score -= 0.05
    if isinstance(orgs, list) and len(orgs) > 0: score -= 0.05
    if isinstance(langs, (list, dict)) and len(langs) > 0: score -= 0.05
    if active_d > 90: score -= 0.05

    # Signaux positifs (=> augmentent la suspicion) : compte bot/inactif
    if repos == 0 and commits == 0:  score += 0.20
    if not bio and not name and not email: score += 0.15
    ratio_ff = following / max(followers, 1)
    if ratio_ff > 20 and followers < 10:  score += 0.15
    if followers == 0 and following > 100: score += 0.10

    return max(0.0, min(1.0, score))


def _build_synthese_instagram(vals: Dict[str, float], importances: np.ndarray, prob_fake: float) -> Dict[str, Any]:
    """
    Synthèse des facteurs pour Instagram/Twitter :
      - chaque feature -> contribution = importance * (déviation par rapport à un profil sain)
      - signaux d'alerte humains
    """
    follower    = vals["userFollowerCount"]
    following   = vals["userFollowingCount"]
    bio_len     = vals["userBiographyLength"]
    posts       = vals["userMediaCount"]
    has_pic     = vals["userHasProfilPic"]
    is_private  = vals["userIsPrivate"]
    digits      = vals["usernameDigitCount"]
    uname_len   = vals["usernameLength"]
    ratio       = follower / (following + 1)

    # Déviation 0..1 par rapport à un "profil sain" attendu
    dev = {
        "userFollowerCount":   1.0 if follower < 20 else (0.5 if follower < 100 else 0.0),
        "userFollowingCount":  1.0 if following > 2000 else (0.4 if following > 1000 else 0.0),
        "userBiographyLength": 1.0 if bio_len < 5 else (0.4 if bio_len < 20 else 0.0),
        "userMediaCount":      1.0 if posts == 0 else (0.4 if posts < 3 else 0.0),
        "userHasProfilPic":    1.0 if has_pic == 0 else 0.0,
        "userIsPrivate":       0.3 if is_private == 1 else 0.0,
        "usernameDigitCount":  min(1.0, digits / 6.0),
        "usernameLength":      1.0 if uname_len > 25 else 0.0,
    }

    contributions = []
    suspicious: List[str] = []
    labels_fr = {
        "userFollowerCount":   "Nombre de followers anormalement bas",
        "userFollowingCount":  "Suit beaucoup trop de comptes",
        "userBiographyLength": "Bio absente ou très courte",
        "userMediaCount":      "Aucune publication / très peu de contenu",
        "userHasProfilPic":    "Pas de photo de profil",
        "userIsPrivate":       "Profil privé",
        "usernameDigitCount":  f"{int(digits)} chiffres dans le pseudo",
        "usernameLength":      "Pseudo anormalement long",
    }

    for i, name in enumerate(INSTAGRAM_FEATURE_ORDER):
        imp = float(importances[i]) if i < len(importances) else 0.0
        deviation = dev[name]
        contrib = round(imp * deviation, 6)
        contributions.append({
            "feature": name,
            "value": vals[name],
            "importance": round(imp, 6),
            "deviation": round(deviation, 3),
            "contribution_to_risk": contrib,
            "explanation": labels_fr[name],
        })
        if deviation >= 0.5:
            suspicious.append(labels_fr[name])

    contributions.sort(key=lambda x: x["contribution_to_risk"], reverse=True)

    return {
        "follower_following_ratio": round(ratio, 4),
        "synthese_facteurs": contributions[:10],
        "signaux_alerte":   suspicious,
        "verdict_humain":   "Profil très suspect" if prob_fake >= 0.7
                            else ("Profil suspect" if prob_fake >= 0.5
                                  else ("À surveiller" if prob_fake >= 0.3 else "Profil cohérent")),
    }


def _build_synthese_linkedin(text_blob: str, prob_fake: float) -> Dict[str, Any]:
    """Synthèse des facteurs pour LinkedIn (basée sur le texte du profil)."""
    blob_lower = text_blob.lower()
    suspicious: List[str] = []
    factors: List[Dict[str, Any]] = []

    n_words = len([w for w in re.findall(r"\w+", blob_lower) if len(w) > 1])
    factors.append({
        "feature": "longueur_texte_profil",
        "value": n_words,
        "explanation": "Volume de texte agrégé du profil (bio + expériences + compétences)",
        "contribution_to_risk": 1.0 if n_words < 20 else (0.3 if n_words < 60 else 0.0),
    })
    if n_words < 20:
        suspicious.append("Profil LinkedIn quasi vide (très peu de contenu textuel)")

    keywords_suspects = [
        "crypto", "bitcoin", "forex", "binary option", "make money", "passive income",
        "investment opportunity", "telegram", "whatsapp only",
    ]
    hits = [k for k in keywords_suspects if k in blob_lower]
    factors.append({
        "feature": "mots_cles_suspects",
        "value": hits,
        "explanation": "Mots-clés à risque détectés dans le profil",
        "contribution_to_risk": min(1.0, len(hits) * 0.4),
    })
    if hits:
        suspicious.append("Mots-clés suspects : " + ", ".join(hits))

    repetition_ratio = 0.0
    words = re.findall(r"\w+", blob_lower)
    if words:
        repetition_ratio = 1.0 - (len(set(words)) / len(words))
    factors.append({
        "feature": "ratio_repetition",
        "value": round(repetition_ratio, 3),
        "explanation": "Profil très répétitif (signal de génération automatique)",
        "contribution_to_risk": min(1.0, max(0.0, (repetition_ratio - 0.6) * 2)),
    })
    if repetition_ratio > 0.75:
        suspicious.append("Texte du profil très répétitif (possible génération automatique)")

    return {
        "synthese_facteurs": factors,
        "signaux_alerte":    suspicious,
        "verdict_humain":    "Profil très suspect" if prob_fake >= 0.7
                             else ("Profil suspect" if prob_fake >= 0.5
                                   else ("À surveiller" if prob_fake >= 0.3 else "Profil cohérent")),
    }


def _build_synthese_bothawk(f: Dict[str, Any], blob: str, prob_fake: float) -> Dict[str, Any]:
    """Synthèse des facteurs pour GitHub (Bothawk)."""
    suspicious: List[str] = []
    factors: List[Dict[str, Any]] = []

    repos    = _safe_int(f.get("num_repository", f.get("public_repos")))
    commits  = _safe_int(f.get("num_commit", f.get("num_commit_estimate")))
    followers= _safe_int(f.get("follower_count"))
    bio      = str(f.get("bio") or "")
    email    = str(f.get("email") or "")
    name     = str(f.get("name") or f.get("display_name") or "")

    def add(feat: str, value: Any, contrib: float, label: str) -> None:
        factors.append({
            "feature": feat, "value": value,
            "contribution_to_risk": round(contrib, 3),
            "explanation": label,
        })
        if contrib >= 0.5:
            suspicious.append(label)

    add("num_repository", repos,
        1.0 if repos == 0 else (0.4 if repos < 2 else 0.0),
        "Aucun dépôt public" if repos == 0 else "Très peu de dépôts publics")
    add("num_commit", commits,
        0.8 if commits == 0 else (0.3 if commits < 5 else 0.0),
        "Aucun commit récent visible" if commits == 0 else "Peu d'activité de commit")
    add("follower_count", followers,
        0.6 if followers == 0 else 0.0,
        "Aucun follower GitHub")
    add("bio_present", bool(bio),
        0.4 if not bio else 0.0,
        "Bio GitHub vide")
    add("email_present", bool(email),
        0.2 if not email else 0.0,
        "Aucun email public")
    add("name_present", bool(name),
        0.3 if not name else 0.0,
        "Nom complet non renseigné")

    return {
        "synthese_facteurs": factors,
        "tokens_bothawk":    blob.split(),
        "signaux_alerte":    suspicious,
        "verdict_humain":    "Compte très suspect" if prob_fake >= 0.7
                             else ("Compte suspect" if prob_fake >= 0.5
                                   else ("À surveiller" if prob_fake >= 0.3 else "Compte cohérent")),
    }


# ----------------------------------------------------------------------------- #
#  PRÉDICTION PAR PLATEFORME                                                     #
# ----------------------------------------------------------------------------- #

def predict_instagram_like(platform: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Prédiction RF Instagram (utilisée aussi pour Twitter/X).

    v7.2 : data-quality gate + hybridation ML/heuristiques.
    """
    load_all_models()
    rf = _models.get("rf_instagram")
    scaler = _models.get("scaler_instagram")
    if rf is None or scaler is None:
        raise RuntimeError(
            f"Modèle Instagram indisponible : "
            f"{_load_errors.get('rf_instagram') or _load_errors.get('scaler_instagram')}"
        )

    X_raw, raw_vals = _instagram_features(features)

    # ----- v7.2 : DATA-QUALITY GATE -----
    is_ok, reason = _instagram_data_quality(raw_vals)
    if not is_ok:
        out = _build_insufficient_data_result(platform, "rf_instagram", raw_vals, reason)
        return out

    X_scaled = scaler.transform(X_raw)
    proba = rf.predict_proba(X_scaled)[0]
    prob_ml = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # ----- v7.2 : HYBRIDATION ML + HEURISTIQUES -----
    prob_heur = _instagram_heuristic_prob(raw_vals)
    # Pondération 50/50 : ne dépend ni d'un dataset biaisé, ni d'heuristiques
    # seules, ce qui équilibre les deux sources de décision.
    prob_fake = 0.50 * prob_ml + 0.50 * prob_heur

    score, label, conf = _normalize_risk(prob_fake)

    importances = getattr(rf, "feature_importances_", np.zeros(len(INSTAGRAM_FEATURE_ORDER)))
    synthese = _build_synthese_instagram(raw_vals, importances, prob_fake)
    synthese["prob_ml"]        = round(prob_ml, 4)
    synthese["prob_heuristic"] = round(prob_heur, 4)
    synthese["prob_final"]     = round(prob_fake, 4)
    synthese["fusion_strategy"] = "50% ML Random Forest + 50% heuristiques métier"

    shap_like = {
        INSTAGRAM_FEATURE_ORDER[i]: round(float(importances[i] * (X_scaled[0, i])), 6)
        for i in range(len(INSTAGRAM_FEATURE_ORDER))
    }

    return {
        "model": "rf_instagram",
        "platform": platform,
        "prediction_score": round(prob_fake, 6),
        "prediction_score_ml":        round(prob_ml,   6),
        "prediction_score_heuristic": round(prob_heur, 6),
        "risk_score": score,
        "classification": label,
        "confidence": conf,
        "metrics": MODEL_PERFORMANCE["rf_instagram"],
        "features": raw_vals,
        "shap_values": shap_like,
        "interpretation": synthese,
    }


def predict_linkedin(features: Dict[str, Any]) -> Dict[str, Any]:
    """Prédiction RF LinkedIn (TF-IDF sur texte agrégé du profil).

    v7.2 : data-quality gate + hybridation ML/heuristiques.
    """
    load_all_models()
    rf = _models.get("rf_linkedin")
    tfidf = _models.get("tfidf_linkedin")
    if rf is None or tfidf is None:
        raise RuntimeError(
            f"Modèle LinkedIn indisponible : "
            f"{_load_errors.get('rf_linkedin') or _load_errors.get('tfidf_linkedin')}"
        )

    blob = _linkedin_text_blob(features)

    # ----- v7.2 : DATA-QUALITY GATE -----
    is_ok, reason = _linkedin_data_quality(blob)
    if not is_ok:
        return _build_insufficient_data_result(
            "linkedin", "rf_linkedin",
            {
                "text_blob_length": len(blob),
                "text_blob_excerpt": blob[:300],
                "word_count": len(re.findall(r"\w+", blob)),
            },
            reason,
        )

    X = tfidf.transform([blob])
    proba = rf.predict_proba(X)[0]
    prob_ml = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # ----- v7.2 : HYBRIDATION ML + HEURISTIQUES -----
    prob_heur = _linkedin_heuristic_prob(blob, features)
    prob_fake = 0.50 * prob_ml + 0.50 * prob_heur

    score, label, conf = _normalize_risk(prob_fake)

    # Top tokens TF-IDF effectivement présents
    vocab_inv = {idx: word for word, idx in tfidf.vocabulary_.items()}
    importances = getattr(rf, "feature_importances_", None)
    top_tokens: List[Dict[str, Any]] = []
    if importances is not None:
        row = X.toarray()[0]
        scored = []
        for idx, val in enumerate(row):
            if val > 0:
                scored.append((vocab_inv.get(idx, f"f_{idx}"), float(importances[idx]) * float(val)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top_tokens = [{"token": t, "weight": round(w, 6)} for t, w in scored[:15]]

    synthese = _build_synthese_linkedin(blob, prob_fake)
    synthese["top_tokens_tfidf"] = top_tokens
    synthese["prob_ml"]        = round(prob_ml, 4)
    synthese["prob_heuristic"] = round(prob_heur, 4)
    synthese["prob_final"]     = round(prob_fake, 4)
    synthese["fusion_strategy"] = "50% ML Random Forest (TF-IDF) + 50% heuristiques métier"

    return {
        "model": "rf_linkedin",
        "platform": "linkedin",
        "prediction_score": round(prob_fake, 6),
        "prediction_score_ml":        round(prob_ml,   6),
        "prediction_score_heuristic": round(prob_heur, 6),
        "risk_score": score,
        "classification": label,
        "confidence": conf,
        "metrics": MODEL_PERFORMANCE["rf_linkedin"],
        "features": {
            "text_blob_length": len(blob),
            "text_blob_excerpt": blob[:300],
            "word_count": len(re.findall(r"\w+", blob)),
        },
        "shap_values": {t["token"]: t["weight"] for t in top_tokens},
        "interpretation": synthese,
    }


def predict_github_bothawk(features: Dict[str, Any]) -> Dict[str, Any]:
    """Prédiction RF GitHub (approche Bothawk : TF-IDF 23 tokens binaires).

    v7.2 : data-quality gate + hybridation forte vers les heuristiques
    (le modèle Bothawk seul est biaisé vers 'fake' à cause de la pauvreté
    du vocabulaire 23 tokens).
    """
    load_all_models()
    rf = _models.get("rf_bothawk")
    tfidf = _models.get("tfidf_bothawk")
    if rf is None or tfidf is None:
        raise RuntimeError(
            f"Modèle GitHub (Bothawk) indisponible : "
            f"{_load_errors.get('rf_bothawk') or _load_errors.get('tfidf_bothawk')}"
        )

    # ----- v7.2 : DATA-QUALITY GATE -----
    is_ok, reason = _bothawk_data_quality(features)
    if not is_ok:
        return _build_insufficient_data_result(
            "github", "rf_bothawk",
            {
                "follower_count":  _safe_int(features.get("follower_count")),
                "following_count": _safe_int(features.get("following_count")),
                "num_repository":  _safe_int(features.get("num_repository",
                                                          features.get("public_repos"))),
                "num_commit":      _safe_int(features.get("num_commit",
                                                          features.get("num_commit_estimate"))),
                "has_bio":         bool(features.get("bio")),
                "has_email":       bool(features.get("email")),
            },
            reason,
        )

    blob = _bothawk_text_blob(features)
    X = tfidf.transform([blob])
    proba = rf.predict_proba(X)[0]
    prob_ml = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # ----- v7.2 : HYBRIDATION (poids plus fort sur l'heuristique pour Bothawk) -----
    prob_heur = _bothawk_heuristic_prob(features)
    # 30 % ML + 70 % heuristique (Bothawk est trop biaisé seul)
    prob_fake = 0.30 * prob_ml + 0.70 * prob_heur

    score, label, conf = _normalize_risk(prob_fake)

    vocab_inv = {idx: word for word, idx in tfidf.vocabulary_.items()}
    importances = getattr(rf, "feature_importances_", None)
    shap_like: Dict[str, float] = {}
    if importances is not None:
        row = X.toarray()[0]
        for idx, val in enumerate(row):
            shap_like[vocab_inv.get(idx, f"f_{idx}")] = round(float(importances[idx]) * float(val), 6)

    synthese = _build_synthese_bothawk(features, blob, prob_fake)
    synthese["prob_ml"]        = round(prob_ml, 4)
    synthese["prob_heuristic"] = round(prob_heur, 4)
    synthese["prob_final"]     = round(prob_fake, 4)
    synthese["fusion_strategy"] = "30% ML Random Forest (Bothawk) + 70% heuristiques métier"

    return {
        "model": "rf_bothawk",
        "platform": "github",
        "prediction_score": round(prob_fake, 6),
        "prediction_score_ml":        round(prob_ml,   6),
        "prediction_score_heuristic": round(prob_heur, 6),
        "risk_score": score,
        "classification": label,
        "confidence": conf,
        "metrics": MODEL_PERFORMANCE["rf_bothawk"],
        "features": {
            "bothawk_tokens": blob,
            "follower_count":   _safe_int(features.get("follower_count")),
            "following_count":  _safe_int(features.get("following_count")),
            "num_repository":   _safe_int(features.get("num_repository", features.get("public_repos"))),
            "num_commit":       _safe_int(features.get("num_commit", features.get("num_commit_estimate"))),
            "has_bio":          bool(features.get("bio")),
            "has_email":        bool(features.get("email")),
        },
        "shap_values": shap_like,
        "interpretation": synthese,
    }


def predict_for_platform(platform: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatcher : route vers le modèle RF approprié selon la plateforme."""
    canonical = (platform or "").strip().lower()
    if canonical not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Plateforme '{platform}' non supportée.")
    model_key = PLATFORM_MODEL_MAP[canonical]
    if model_key == "rf_instagram":
        return predict_instagram_like(canonical, features)
    if model_key == "rf_linkedin":
        return predict_linkedin(features)
    if model_key == "rf_bothawk":
        return predict_github_bothawk(features)
    raise RuntimeError(f"Aucun modèle mappé pour la plateforme : {canonical}")


# ----------------------------------------------------------------------------- #
#  MODULE JOB : DÉTECTION D'OFFRES D'EMPLOI FRAUDULEUSES (LSTM + heuristiques)   #
# ----------------------------------------------------------------------------- #

# Mots-clés / patterns classiques des arnaques d'offres d'emploi
SUSPICIOUS_PATTERNS_JOB: List[Tuple[str, float, str]] = [
    (r"urgent(?:ly)?",                 0.10, "Insistance sur l'urgence"),
    (r"telegram",                      0.12, "Communication via Telegram"),
    (r"whatsapp",                      0.10, "Communication via WhatsApp uniquement"),
    (r"wire\s+transfer",               0.18, "Demande de virement bancaire"),
    (r"registration\s+fee",            0.20, "Frais d'inscription demandés"),
    (r"upfront(?:\s+payment)?",        0.18, "Paiement en amont demandé"),
    (r"crypto|bitcoin|usdt|eth\b",     0.15, "Mention de cryptomonnaie"),
    (r"no\s+experience\s+(?:required|needed)", 0.08, "Aucune expérience requise"),
    (r"limited\s+slots?",              0.07, "Places limitées (pression)"),
    (r"apply\s+now",                   0.04, "Pression à postuler immédiatement"),
    (r"immediate\s+start",             0.06, "Démarrage immédiat"),
    (r"data\s+entry",                  0.06, "Saisie de données (arnaque classique)"),
    (r"work\s+from\s+home\s+only",     0.07, "100% télétravail sans entretien"),
    (r"guaranteed\s+income",           0.12, "Revenu garanti"),
    (r"earn\s+\$?\d{3,}\s+(?:per|/)\s*(?:day|week|hr|hour)", 0.18,
                                       "Promesse de gains élevés à la journée/heure"),
    (r"personal\s+(?:bank|account)\s+details", 0.20, "Demande de coordonnées bancaires"),
    (r"send\s+(?:your\s+)?(?:cv|resume)\s+to\s+\S+@", 0.04,
                                       "Candidature uniquement par e-mail personnel"),
]
URL_REGEX = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PUBLIC_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                        "protonmail.com", "icloud.com", "live.com"}


def _lstm_preprocess(text: str):
    """Tokenisation + padding pour le LSTM (implémentation pure Python)."""
    seq = _lstm_tokenizer.texts_to_sequences([text])
    return _lite_pad_sequences(seq, maxlen=LSTM_MAX_SEQUENCE_LEN, dtype="float32")


def analyze_job_signals(text: str) -> Dict[str, Any]:
    """Analyse heuristique des signaux frauduleux dans un texte d'offre d'emploi."""
    if not text:
        return {}
    lowered = text.lower()
    matched: List[Dict[str, Any]] = []
    heuristic_bonus = 0.0
    for pat, weight, label in SUSPICIOUS_PATTERNS_JOB:
        if re.search(pat, lowered, flags=re.IGNORECASE):
            matched.append({"pattern": pat, "label": label, "weight": weight})
            heuristic_bonus += weight

    urls   = URL_REGEX.findall(text)
    emails = EMAIL_REGEX.findall(text)
    public_email_count = 0
    for e in emails:
        domain = e.split("@")[-1].lower()
        if domain in PUBLIC_EMAIL_DOMAINS:
            public_email_count += 1
    if public_email_count > 0:
        heuristic_bonus += 0.08 * public_email_count
        matched.append({
            "pattern": "public_email_domain",
            "label":   f"{public_email_count} adresse(s) e-mail sur domaine public (gmail/yahoo/...)",
            "weight":  0.08 * public_email_count,
        })

    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    upper_ratio = round(upper / alpha, 4) if alpha else 0.0
    if upper_ratio > 0.30:
        heuristic_bonus += 0.05
        matched.append({"pattern": "shouty_text", "label": "Texte criard (ratio majuscules anormal)", "weight": 0.05})

    tokens = re.findall(r"\b\w+\b", text)
    return {
        "text_length": len(text),
        "token_count": len(tokens),
        "url_count":   len(urls),
        "email_count": len(emails),
        "public_email_count": public_email_count,
        "uppercase_ratio": upper_ratio,
        "suspicious_keyword_count": len(matched),
        "suspicious_keywords": [m["label"] for m in matched[:12]],
        "heuristic_bonus_score": round(min(heuristic_bonus, 0.5), 3),
        "matched_patterns": matched[:12],
    }


def predict_job_text(text: str) -> Dict[str, Any]:
    """
    Combine LSTM (60%) + heuristiques (40%) pour produire un score final
    de fraude d'offre d'emploi.
    """
    load_lstm()
    if _lstm_error or _lstm_interpreter is None:
        raise RuntimeError(
            f"Modèle LSTM offres d'emploi indisponible : {_lstm_error or 'non chargé'}"
        )

    # 1. Prédiction LSTM
    inp = _lstm_preprocess(text)
    _lstm_interpreter.set_tensor(_lstm_in[0]["index"], inp)
    _lstm_interpreter.invoke()
    lstm_score = float(_lstm_interpreter.get_tensor(_lstm_out[0]["index"])[0][0])
    lstm_score = max(0.0, min(1.0, lstm_score))

    # 2. Heuristiques
    signals = analyze_job_signals(text)
    heuristic_bonus = signals.get("heuristic_bonus_score", 0.0)

    # 3. Combinaison pondérée
    final_prob = max(0.0, min(1.0, 0.6 * lstm_score + 0.4 * min(1.0, lstm_score + heuristic_bonus)))
    score, label, conf = _normalize_risk(final_prob)

    # Synthèse facteurs pour le job
    synthese_facteurs = [
        {
            "feature": "lstm_prediction",
            "value": round(lstm_score, 6),
            "contribution_to_risk": round(lstm_score, 3),
            "explanation": "Probabilité LSTM (modèle entraîné sur les offres frauduleuses)",
        },
        {
            "feature": "patterns_suspects",
            "value": signals.get("suspicious_keyword_count", 0),
            "contribution_to_risk": round(heuristic_bonus, 3),
            "explanation": "Mots-clés / patterns d'arnaque détectés",
        },
        {
            "feature": "url_count",
            "value": signals.get("url_count", 0),
            "contribution_to_risk": round(min((signals.get("url_count", 0) - 2) * 0.1, 0.4), 3) if signals.get("url_count", 0) > 2 else 0.0,
            "explanation": "Nombre de liens dans l'offre",
        },
        {
            "feature": "uppercase_ratio",
            "value": signals.get("uppercase_ratio", 0.0),
            "contribution_to_risk": round(min(signals.get("uppercase_ratio", 0.0) * 1.0, 0.3), 3),
            "explanation": "Texte excessivement en majuscules",
        },
        {
            "feature": "public_email_count",
            "value": signals.get("public_email_count", 0),
            "contribution_to_risk": round(min(signals.get("public_email_count", 0) * 0.15, 0.3), 3),
            "explanation": "Contact via boîtes mail publiques (gmail/yahoo/...)",
        },
    ]
    synthese_facteurs.sort(key=lambda x: x["contribution_to_risk"], reverse=True)

    return {
        "model": "job_text_lstm",
        "platform": "job",
        "prediction_score": round(final_prob, 6),
        "lstm_raw_score":   round(lstm_score, 6),
        "heuristic_bonus":  heuristic_bonus,
        "risk_score": score,
        "classification": label,
        "confidence": conf,
        "threshold": LSTM_THRESHOLD,
        "metrics": MODEL_PERFORMANCE["job_text_lstm"],
        "signals": signals,
        "features": {
            "input_mode": "job_text",
            "job_text_excerpt": re.sub(r"\s+", " ", text).strip()[:280],
            "job_text_length": signals.get("text_length", 0),
            "token_count": signals.get("token_count", 0),
            "url_count": signals.get("url_count", 0),
            "email_count": signals.get("email_count", 0),
            "uppercase_ratio": signals.get("uppercase_ratio", 0.0),
            "suspicious_keyword_count": signals.get("suspicious_keyword_count", 0),
            "suspicious_keywords": signals.get("suspicious_keywords", []),
            "prediction_score": round(final_prob, 6),
            "threshold": LSTM_THRESHOLD,
        },
        "shap_values": {
            "lstm_raw":         round(lstm_score - 0.5, 3),
            "suspicious_count": round(min(signals.get("suspicious_keyword_count", 0) * 0.08, 0.4), 3),
            "url_count":        round(min(signals.get("url_count", 0) * 0.07, 0.21), 3),
            "email_count":      round(min(signals.get("email_count", 0) * 0.05, 0.15), 3),
            "uppercase_ratio":  round(signals.get("uppercase_ratio", 0.0) * 0.5, 3),
        },
        "interpretation": {
            "synthese_facteurs": synthese_facteurs,
            "signaux_alerte":    signals.get("suspicious_keywords", []),
            "verdict_humain": "Offre très probablement frauduleuse" if final_prob >= 0.7
                              else ("Offre suspecte" if final_prob >= 0.5
                                    else ("À vérifier" if final_prob >= 0.3 else "Offre crédible")),
        },
    }



# ----------------------------------------------------------------------------- #
#  Orchestration d'extraction                                                    #
# ----------------------------------------------------------------------------- #

def build_ml_features(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Construit la couche commune de features ML à partir du profil extrait."""
    username = str(profile.get("username") or "")
    bio = str(profile.get("bio") or "")
    follower = _safe_int(profile.get("follower_count"))
    following = _safe_int(profile.get("following_count"))
    post = _safe_int(profile.get("post_count"))
    return {
        # Communs
        "follower_count": follower,
        "following_count": following,
        "post_count": post,
        "bio_length": len(bio),
        "username_length": len(username),
        "username_digits": _count_digits(username),
        "has_profile_pic": bool(profile.get("avatar_url")),
        "is_private": bool(profile.get("is_private", False)),
        "follower_following_ratio": round(follower / (following + 1), 4),
        # Instagram-style
        "userFollowerCount":   follower,
        "userFollowingCount":  following,
        "userMediaCount":      post,
        "userBiographyLength": len(bio),
        "usernameLength":      len(username),
        "usernameDigitCount":  _count_digits(username),
        "userHasProfilPic":    bool(profile.get("avatar_url")),
        "userIsPrivate":       bool(profile.get("is_private", False)),
        # LinkedIn-style (texte agrégé)
        "bio":         profile.get("bio") or "",
        "name":        profile.get("display_name") or "",
        "headline":    profile.get("headline") or "",
        "company":     profile.get("company") or "",
        "industry":    profile.get("industry") or "",
        "location":    profile.get("location") or "",
        "skills":      profile.get("skills") or [],
        "experiences": profile.get("experiences") or [],
        "educations":  profile.get("educations") or [],
        # GitHub-style
        "num_repository":      _safe_int(profile.get("public_repos")),
        "num_commit":          _safe_int(profile.get("num_commit",
                                                     profile.get("num_commit_estimate"))),
        "num_activities":      _safe_int(profile.get("num_activities")),
        "active_days":         _safe_int(profile.get("active_days")),
        "email":               profile.get("email") or "",
        "login":               profile.get("username") or "",
        "organizations":       profile.get("organizations") or [],
        "languages":           profile.get("languages") or [],
    }


def run_extraction(url: str, forced_platform: Optional[str] = None) -> Dict[str, Any]:
    """Détecte la plateforme et lance l'extracteur correspondant (avec fallback générique)."""
    platform, detection = (forced_platform, "forced") if forced_platform else detect_platform(url)
    if platform not in EXTRACTORS:
        platform, detection = "generic", "unknown_forced_fallback"

    t0 = time.time()
    try:
        profile = EXTRACTORS[platform](url)
    except ValueError as exc:
        return {"success": False, "platform": platform, "detected_via": detection,
                "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        if platform != "generic":
            try:
                profile = extract_generic(url)
                profile["_fallback_from"]   = platform
                profile["_fallback_reason"] = str(exc)
                platform, detection = "generic", detection + "+fallback"
            except Exception as exc2:  # noqa: BLE001
                return {"success": False, "platform": platform, "detected_via": detection,
                        "message": f"Extraction impossible : {exc2}"}
        else:
            return {"success": False, "platform": platform, "detected_via": detection,
                    "message": f"Extraction impossible : {exc}"}

    elapsed_ms = int((time.time() - t0) * 1000)
    response: Dict[str, Any] = {
        "success": True, "platform": platform, "detected_via": detection,
        "elapsed_ms": elapsed_ms, "profile": profile, "source_url": url,
        "extraction_method":  profile.get("extraction_method"),
        "extraction_warning": profile.get("extraction_warning"),
    }
    if profile.get("is_job_posting"):
        response["job_text"] = profile.get("job_text", "")
        response["features"] = None
    else:
        response["features"] = build_ml_features(profile)
    return response


# ----------------------------------------------------------------------------- #
#  Routes Flask                                                                  #
# ----------------------------------------------------------------------------- #

def _gen_analysis_id() -> str:
    return "AN-" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + str(np.random.randint(1000, 9999))


@app.route("/")
def home():
    return jsonify({
        "service": "FakeGuard / FPD API",
        "version": "7.2.0",
        "thresholds": {
            "fake_threshold":       FAKE_THRESHOLD,
            "suspicious_threshold": SUSPICIOUS_THRESHOLD,
        },
        "status":  "active",
        "models_active": list(MODEL_PERFORMANCE.keys()),
        "supported_platforms": sorted(SUPPORTED_PLATFORMS),
        "platform_model_map": PLATFORM_MODEL_MAP,
        "lstm_loaded": _lstm_interpreter is not None,
        "rf_models_loaded": [m for m in _models
                             if not m.startswith("scaler") and not m.startswith("tfidf")],
        "endpoints": {
            "GET  /health":          "Healthcheck",
            "GET  /api/models":      "Liste des modèles ML",
            "POST /api/analyze":     "Analyse d'un profil (features manuelles)",
            "POST /api/analyze-job": "Analyse LSTM d'un texte d'offre d'emploi",
            "POST /api/extract-url": "Extraction brute depuis une URL",
            "POST /api/analyze-url": "Extraction + analyse complète depuis une URL",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "version": "7.2.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "lstm_loaded": _lstm_interpreter is not None,
        "rf_loaded": {
            "rf_instagram": "rf_instagram" in _models,
            "rf_linkedin":  "rf_linkedin"  in _models,
            "rf_bothawk":   "rf_bothawk"   in _models,
        },
        "load_errors": _load_errors,
        "lstm_error":  _lstm_error,
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify({
        "success": True,
        "models": MODEL_PERFORMANCE,
        "supported_platforms": sorted(SUPPORTED_PLATFORMS),
        "platform_model_map":  PLATFORM_MODEL_MAP,
    })


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_profile():
    """Analyse d'un profil avec features manuelles."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        platform = (data.get("platform") or "").strip().lower()
        features = data.get("features") or {}

        if not platform:
            return jsonify({"success": False, "message": "Plateforme manquante."}), 422
        if platform not in SUPPORTED_PLATFORMS:
            return jsonify({"success": False,
                            "message": f"Plateforme '{platform}' non prise en charge."}), 422
        if not isinstance(features, dict) or not features:
            return jsonify({"success": False, "message": "Caractéristiques manquantes."}), 422

        result = predict_for_platform(platform, features)
        return jsonify({
            "success": True,
            "analysis_id": _gen_analysis_id(),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            **result,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


@app.route("/api/analyze-job", methods=["POST", "OPTIONS"])
def analyze_job():
    """Analyse LSTM + heuristiques d'un texte d'offre d'emploi."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("job_text") or data.get("text") or "").strip()
        if not text:
            return jsonify({"success": False,
                            "message": "Le texte de l'offre d'emploi est obligatoire."}), 422
        if len(text) < 30:
            return jsonify({"success": False,
                            "message": "Le texte saisi est trop court pour une détection fiable."}), 422

        result = predict_job_text(text)
        return jsonify({
            "success": True,
            "analysis_id": _gen_analysis_id(),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            **result,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


@app.route("/api/extract-url", methods=["POST", "OPTIONS"])
def extract_url_endpoint():
    """Extraction brute (pas de scoring) à partir d'une URL."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        forced = (data.get("platform") or "").strip().lower() or None
        if not url:
            return jsonify({"success": False, "message": "Le champ 'url' est obligatoire."}), 422
        if not re.match(r"^https?://", url):
            url = "https://" + url

        ok, reason = _is_safe_public_url(url)
        if not ok:
            return jsonify({"success": False, "message": f"URL refusée : {reason}"}), 422

        result = run_extraction(url, forced_platform=forced)

        # ---------------------------------------------------------------- #
        # FIX v7.1.1 : expose les compteurs canoniques au plus haut niveau #
        # afin que tout front-end puisse les afficher immédiatement,        #
        # indépendamment de la plateforme.                                 #
        # ---------------------------------------------------------------- #
        if result.get("success"):
            prof  = result.get("profile")  or {}
            feats = result.get("features") or {}
            f_cnt = _safe_int(prof.get("follower_count",
                                       feats.get("follower_count", 0)))
            fo_cnt = _safe_int(prof.get("following_count",
                                        feats.get("following_count", 0)))
            p_cnt = _safe_int(prof.get("post_count",
                                       feats.get("post_count", 0)))
            # Assure la présence des clés dans le profil
            prof["follower_count"]  = f_cnt
            prof["following_count"] = fo_cnt
            prof["post_count"]      = p_cnt
            result["profile"] = prof
            # Et au plus haut niveau (lecture directe par le front)
            result["follower_count"]  = f_cnt
            result["following_count"] = fo_cnt
            result["post_count"]      = p_cnt

        return jsonify(result), (200 if result.get("success") else 502)

    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


@app.route("/api/analyze-url", methods=["POST", "OPTIONS"])
def analyze_url_endpoint():
    """Extraction + scoring complet depuis une URL (profil ou offre d'emploi)."""
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        forced = (data.get("platform") or "").strip().lower() or None

        if not url:
            return jsonify({"success": False, "message": "URL manquante."}), 422
        if not re.match(r"^https?://", url):
            url = "https://" + url

        ok, reason = _is_safe_public_url(url)
        if not ok:
            return jsonify({"success": False, "message": f"URL refusée : {reason}"}), 422

        extraction = run_extraction(url, forced_platform=forced)
        if not extraction.get("success"):
            return jsonify({
                "success": False,
                "message":  extraction.get("message", "Extraction impossible."),
                "platform": extraction.get("platform"),
            }), 502

        platform = extraction.get("platform", "generic")
        profile  = extraction.get("profile") or {}

        # ----- Cas offre d'emploi -> LSTM -----
        if platform == "job" or profile.get("is_job_posting"):
            job_text = extraction.get("job_text") or profile.get("job_text") or ""
            if len(job_text) < 30:
                return jsonify({"success": False,
                                "message": "Contenu de l'offre extraite trop court."}), 422
            res = predict_job_text(job_text)
            feats = res["features"]
            feats.update({
                "input_mode": "url_job",
                "source_url": url,
                "job_title":           profile.get("title"),
                "hiring_organization": profile.get("hiring_organization"),
                "job_location":        profile.get("job_location"),
                "contact_email":       profile.get("contact_email"),
            })
            return jsonify({
                "success": True,
                "analysis_id": _gen_analysis_id(),
                "timestamp":   datetime.utcnow().isoformat() + "Z",
                "platform":    "job",
                "detected_via": extraction.get("detected_via"),
                "extraction_method":  extraction.get("extraction_method"),
                "extraction_warning": extraction.get("extraction_warning"),
                "source_url":  url,
                "model":       res["model"],
                "risk_score":  res["risk_score"],
                "classification": res["classification"],
                "confidence":  res["confidence"],
                "metrics":     res["metrics"],
                "shap_values": res["shap_values"],
                "features":    feats,
                "interpretation":   res["interpretation"],
                "extracted_profile": profile,
            }), 200

        # ----- Cas profil -> RF (instagram/twitter/linkedin/github) -----
        if platform not in SUPPORTED_PLATFORMS:
            return jsonify({
                "success": False,
                "message": f"Plateforme '{platform}' non prise en charge.",
                "platform": platform,
            }), 422

        raw_features = extraction.get("features") or {}
        res = predict_for_platform(platform, raw_features)

        # ---------------------------------------------------------------- #
        # FIX v7.1.1 : on garantit que les compteurs canoniques            #
        # (follower_count / following_count / post_count) sont TOUJOURS    #
        # présents dans `features` et dans `extracted_profile`, même       #
        # quand le predicteur ne les ré-expose pas (LinkedIn renvoie un    #
        # text_blob, Instagram renvoie des clés camelCase, etc.).          #
        # On s'appuie sur `raw_features` (issu de build_ml_features) et    #
        # sur `profile` (issu de l'extracteur de plateforme).              #
        # ---------------------------------------------------------------- #
        canonical_follower  = _safe_int(
            profile.get("follower_count",
                        raw_features.get("follower_count",
                                         raw_features.get("userFollowerCount", 0)))
        )
        canonical_following = _safe_int(
            profile.get("following_count",
                        raw_features.get("following_count",
                                         raw_features.get("userFollowingCount", 0)))
        )
        canonical_posts     = _safe_int(
            profile.get("post_count",
                        raw_features.get("post_count",
                                         raw_features.get("userMediaCount", 0)))
        )

        # On enrichit aussi le `profile` retourné au front pour qu'il      #
        # contienne systématiquement les trois compteurs.                  #
        profile.setdefault("follower_count",  canonical_follower)
        profile.setdefault("following_count", canonical_following)
        profile.setdefault("post_count",      canonical_posts)
        # Et on force les valeurs si elles existaient mais étaient None.
        if profile.get("follower_count")  in (None, ""): profile["follower_count"]  = canonical_follower
        if profile.get("following_count") in (None, ""): profile["following_count"] = canonical_following
        if profile.get("post_count")      in (None, ""): profile["post_count"]      = canonical_posts

        merged_features = {
            "input_mode": "url",
            "source_url": url,
            "platform_detected_via": extraction.get("detected_via"),
            "username":     profile.get("username"),
            "display_name": profile.get("display_name"),
            "bio_excerpt":  (profile.get("bio") or "")[:240],
            "avatar_url":   profile.get("avatar_url"),
            **(res["features"] if isinstance(res["features"], dict)
               else {"features": res["features"]}),
            # === Compteurs canoniques (toujours présents) === #
            "follower_count":   canonical_follower,
            "following_count":  canonical_following,
            "post_count":       canonical_posts,
            # Alias camelCase Instagram-style (pour compat. front)          #
            "userFollowerCount":  canonical_follower,
            "userFollowingCount": canonical_following,
            "userMediaCount":     canonical_posts,
        }

        return jsonify({
            "success": True,
            "analysis_id": _gen_analysis_id(),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "platform":    platform,
            "detected_via": extraction.get("detected_via"),
            "extraction_method":  extraction.get("extraction_method"),
            "extraction_warning": extraction.get("extraction_warning"),
            "source_url":  url,
            "model":       res["model"],
            "risk_score":  res["risk_score"],
            "classification": res["classification"],
            "confidence":  res["confidence"],
            "metrics":     res["metrics"],
            "shap_values": res["shap_values"],
            "features":    merged_features,
            "interpretation":   res["interpretation"],
            "extracted_profile": profile,
            # Champs raccourcis exposés au plus haut niveau (pratique       #
            # pour les fronts qui lisent directement la racine).            #
            "follower_count":   canonical_follower,
            "following_count":  canonical_following,
            "post_count":       canonical_posts,
        }), 200

    except RuntimeError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "message": f"Erreur serveur : {exc}"}), 500


# ----------------------------------------------------------------------------- #
#  Erreurs globales                                                              #
# ----------------------------------------------------------------------------- #

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"success": False, "message": "Endpoint introuvable."}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"success": False, "message": "Méthode non autorisée."}), 405


# ----------------------------------------------------------------------------- #
#  Démarrage                                                                     #
# ----------------------------------------------------------------------------- #

# Préchargement opportuniste au démarrage
load_all_models()
load_lstm()

if __name__ == "__main__":
    print(f"[FakeGuard] Démarrage sur http://0.0.0.0:{PORT}", flush=True)
    print(f"  RF Instagram   : {'rf_instagram' in _models}", flush=True)
    print(f"  RF LinkedIn    : {'rf_linkedin'  in _models}", flush=True)
    print(f"  RF Bothawk(GH) : {'rf_bothawk'   in _models}", flush=True)
    print(f"  LSTM job       : {_lstm_interpreter is not None}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False)
