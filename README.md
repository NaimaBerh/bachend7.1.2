# Backend FakeGuard / FPD — v7.1

Backend **Python / Flask** pour le détecteur de faux profils & d'offres
d'emploi frauduleuses. Compatible avec le frontend **FPD-frontend-v7**.

> **v7.1** : extracteurs réécrits pour résoudre le problème d'extraction
> incomplète des informations depuis Instagram, Twitter/X et LinkedIn.

## ✨ Nouveautés v7.1

| Plateforme | Méthode v7.0 (avant) | Méthode v7.1 (après) |
|---|---|---|
| **Instagram** | OG description + regex EN uniquement (échec quasi-systématique) | Endpoint public `i.instagram.com/api/v1/users/web_profile_info` (JSON complet) **+** fallback OG **+** scan JSON inliné **+** regex FR/EN/ES |
| **Twitter/X** | OG seul (X bloque tout sans JS → vide) | `syndication.twitter.com` (HTML+JSON) **+** `cdn.syndication.twimg.com` (JSON) **+** fallback Nitter en cascade |
| **LinkedIn** | OG + JSON-LD (souvent absent sur authwall) | oEmbed officiel **+** OG/Twitter card **+** JSON-LD **+** microdata `itemprop` **+** scan JSON inliné |
| **GitHub** | API officielle + estimation commits via /events | Idem **+** Search API pour commits réels (bornée) **+** orgs **+** langages **+** active_days |
| **Offres d'emploi** | JSON-LD JobPosting uniquement | JSON-LD **+** microdata HTML **+** fallback texte brut + 4 User-Agents en rotation |

## 🔧 Améliorations techniques

- **Rotation User-Agent** : 4 UAs (Chrome desktop, Safari macOS, Firefox Linux, Safari iOS)
- **Retries automatiques** sur 403/429 avec UA différent
- **En-têtes anti-bot réalistes** : `Sec-Fetch-*`, `Accept-Language fr-FR`, `Upgrade-Insecure-Requests`...
- **Champ `extraction_method`** retourné dans toutes les réponses : décrit la stratégie qui a fonctionné
- **Champ `extraction_warning`** : alerte l'utilisateur quand seule une partie des données a été récupérée
- **Endpoint `/health`** affiche désormais `"version": "7.1.0"`

## 📊 Ce qui est implémenté

| Plateforme        | Modèle utilisé                                  | Approche                                              |
|-------------------|-------------------------------------------------|-------------------------------------------------------|
| **Instagram**     | `rf_instagram.pkl` + `scaler_instagram.pkl`     | Random Forest sur 8 features numériques               |
| **Twitter / X**   | `rf_instagram.pkl` + `scaler_instagram.pkl`     | Réutilisé (profil grand-public similaire)             |
| **LinkedIn**      | `rf_linkedin.pkl` + `tfidf_linkedin.pkl`        | TF-IDF (1000 mots) sur le texte agrégé du profil      |
| **GitHub**        | `rf_bothawk.pkl` + `tfidf_bothawk.pkl`          | Approche **Bothawk** (TF-IDF de 23 tokens-signaux)    |
| **Offre d'emploi**| `fake_job_lstm_model.tflite` + `tokenizer.json` | LSTM (60%) + heuristiques anti-arnaque (40%)          |

Chaque réponse expose un **score de risque (`risk_score` 0–100)**,
une **classification** (`fake` / `genuine`), une **confiance**, et surtout
une **`interpretation.synthese_facteurs`** qui détaille la contribution de
chaque facteur au score.

## 📁 Structure

```
backend/
├── app.py                   # Toute la logique backend (un seul fichier)
├── requirements.txt
├── .env.example
├── README.md
├── run.sh                   # Lancement Linux/macOS
├── run.bat                  # Lancement Windows
└── models/
    ├── rf_instagram.pkl
    ├── scaler_instagram.pkl
    ├── rf_linkedin.pkl
    ├── tfidf_linkedin.pkl
    ├── rf_bothawk.pkl
    ├── tfidf_bothawk.pkl
    ├── fake_job_lstm_model.tflite
    └── tokenizer.json
```

## 🚀 Installation

### 1. Prérequis
- **Python 3.10 ou 3.11** (TensorFlow 2.16 ne supporte pas encore 3.13)
- pip à jour

### 2. Création de l'environnement virtuel

**Linux / macOS**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration (optionnelle)
```bash
cp .env.example .env
# puis éditez .env si nécessaire (port, token GitHub, etc.)
```

### 4. Lancement

**Linux / macOS**
```bash
./run.sh
# ou directement :
python app.py
```

**Windows**
```cmd
run.bat
```

L'API écoute par défaut sur `http://127.0.0.1:5000`.

### 5. Vérification rapide
```bash
curl http://127.0.0.1:5000/health
```
Vous devez voir tous les modèles à `true` et `"version": "7.1.0"`.

## 🔌 Branchement avec le frontend

Le fichier PHP `api/flask_config.php` du frontend pointe déjà sur
`http://127.0.0.1:5000`. Aucune modification n'est nécessaire pour les
tests locaux.

## 🔒 Sécurité de l'extraction d'URL

L'extracteur applique plusieurs barrières pour empêcher les attaques **SSRF**
et le scraping abusif :

- Schémas autorisés : `http`, `https` uniquement
- Résolution DNS + **rejet des IP internes** (`127.0.0.0/8`, `10.0.0.0/8`,
  `172.16.0.0/12`, `192.168.0.0/16`, link-local, CGNAT, IPv6 ULA, etc.)
- Rejet de `localhost`, `*.local`, `*.internal`
- **Timeout strict** (15 s) et **limite de redirections** (5)
- **Limite de taille de réponse** (4 Mo) pour éviter les memory bombs
- En-têtes HTTP réalistes (rotation 4 User-Agents)
- Pour GitHub on **privilégie l'API officielle** `api.github.com`
- Pour Instagram on **privilégie l'endpoint** `web_profile_info` (JSON stable)
- Pour Twitter on **privilégie** `syndication.twitter.com` (sans authentification)

## 📡 Endpoints

| Méthode | URL                  | Description                                                     |
|---------|----------------------|-----------------------------------------------------------------|
| GET     | `/`                  | Informations API                                                |
| GET     | `/health`            | État de chargement de tous les modèles                          |
| GET     | `/api/models`        | Liste des modèles ML + mapping plateforme→modèle                |
| POST    | `/api/analyze`       | Analyse d'un profil avec features manuelles                     |
| POST    | `/api/analyze-job`   | Analyse d'un texte d'offre d'emploi (LSTM + heuristiques)       |
| POST    | `/api/extract-url`   | Extraction brute des informations d'un profil/offre depuis URL  |
| POST    | `/api/analyze-url`   | Extraction + scoring complet depuis URL                         |

Toutes les réponses d'extraction incluent maintenant :
- `extraction_method` : la stratégie qui a effectivement fonctionné
  (ex. `instagram_web_profile_info_api`, `twitter_syndication_html`,
  `linkedin_oembed+html_og+jsonld+microdata`, `github_official_api`)
- `extraction_warning` : présent quand seule une partie des données a pu
  être récupérée (par ex. quand LinkedIn renvoie un authwall)

### Exemple : `POST /api/analyze`
```json
{
  "platform": "instagram",
  "features": {
    "userFollowerCount": 12,
    "userFollowingCount": 1500,
    "userMediaCount": 0,
    "userBiographyLength": 0,
    "usernameLength": 18,
    "usernameDigitCount": 6,
    "userHasProfilPic": false,
    "userIsPrivate": false
  }
}
```

### Exemple : `POST /api/analyze-url`
```json
{ "url": "https://github.com/torvalds" }
```

### Exemple : `POST /api/analyze-job`
```json
{
  "job_text": "Urgent! Work from home, earn $500 per day. Contact me on Telegram, no experience needed. Send your CV to jane.smith@gmail.com. Limited slots, apply now!"
}
```

## 🧪 Tests rapides

```bash
# Healthcheck
curl http://127.0.0.1:5000/health | python -m json.tool

# Analyse d'un profil Instagram suspect
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"platform":"instagram","features":{"userFollowerCount":3,"userFollowingCount":2500,"userMediaCount":0,"userBiographyLength":0,"usernameLength":22,"usernameDigitCount":8,"userHasProfilPic":false,"userIsPrivate":false}}'

# Extraction d'une URL Instagram (test la chaîne complète)
curl -X POST http://127.0.0.1:5000/api/extract-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.instagram.com/instagram/"}'

# Extraction d'un profil Twitter/X
curl -X POST http://127.0.0.1:5000/api/extract-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://x.com/elonmusk"}'

# Analyse d'un compte GitHub par URL
curl -X POST http://127.0.0.1:5000/api/analyze-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://github.com/torvalds"}'

# Analyse d'une offre d'emploi suspecte
curl -X POST http://127.0.0.1:5000/api/analyze-job \
  -H "Content-Type: application/json" \
  -d '{"job_text":"Urgent! Earn $500/day from home. Contact on Telegram. No experience needed. Limited slots, apply now!"}'
```

## ⚠️ Limites connues du scraping public

Instagram, X.com et LinkedIn évoluent constamment pour bloquer le scraping
non-authentifié. **Le backend ne peut pas se connecter avec un compte
utilisateur** (cela violerait leurs CGU), il s'appuie donc uniquement sur
les endpoints publics et les balises OpenGraph/JSON-LD.

Quand une plateforme bloque tout (par ex. LinkedIn affiche un authwall depuis
une IP de datacenter), la réponse contient un champ `extraction_warning`
qui le signale clairement, et seules les données fragmentaires disponibles
sont retournées. Le scoring ML continue de tourner sur les features
disponibles, mais sa fiabilité est mécaniquement plus faible.

**Bonnes pratiques côté production** :
- Faire tourner le backend depuis une IP résidentielle (pas un datacenter
  cloud) pour éviter les blocages préventifs.
- Renseigner `GITHUB_TOKEN` dans `.env` pour passer la limite GitHub
  de 60 → 5000 requêtes/heure.
- Pour la collecte massive : utiliser un proxy résidentiel avec rotation.

## 💡 Astuces

- **Alléger TensorFlow** : pour économiser de la RAM, remplacez `tensorflow`
  par `tflite-runtime` dans `requirements.txt` et adaptez les 2 imports
  `tf.lite.Interpreter` / `tokenizer_from_json` dans `app.py`.
- **Quota GitHub** : sans `GITHUB_TOKEN`, l'API publique est limitée à
  60 requêtes/heure par IP. Avec un token, on passe à 5000 req/h.
- **Production** : utilisez `gunicorn -w 2 -b 0.0.0.0:5000 app:app`
  (le `requirements.txt` inclut déjà gunicorn).
