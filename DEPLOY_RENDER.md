# 🚀 Déploiement sur Render — FakeGuard / FPD Backend

Ce dossier est **prêt à être déployé** sur [Render](https://render.com) sans aucune modification supplémentaire.

---

## 📋 Résumé des adaptations effectuées

| Changement | Avant | Après | Raison |
|---|---|---|---|
| `tensorflow` | 2.16.2 (~500 Mo) | **`tflite-runtime` 2.14.0** (~3 Mo) | Le free tier Render limite le slug à ~500 Mo |
| Tokenizer Keras | `tf.keras.preprocessing.text.tokenizer_from_json` | **Réimplémentation pure Python** (`_LiteKerasTokenizer`) | Élimine la dépendance TF |
| `pad_sequences` | `tf.keras...pad_sequences` | **Réimplémentation NumPy** (`_lite_pad_sequences`) | Élimine la dépendance TF |
| Serveur | `python app.py` (Flask dev) | **Gunicorn 22** (`--preload`, 4 threads) | Production-grade |
| Configuration | `.env` manuel | **`render.yaml`** + `Procfile` | Provisionnement automatique |
| Structure | `backend/` imbriqué | **Racine du dépôt** | Détection automatique par Render |
| Healthcheck | absent | **`/health`** déclaré | Render redémarre auto en cas de panne |

Le **comportement métier est identique** : mêmes endpoints, mêmes modèles, mêmes scores. Seule la chaîne d'inférence LSTM a été allégée (et reste exacte, car le tokenizer Keras est entièrement décrit en JSON et a été ré-implémenté à l'identique).

---

## 🪜 Étapes de déploiement (5 minutes)

### Étape 1 — Pousser le projet sur GitHub

```bash
cd FPD-backend-render
git init
git add .
git commit -m "Initial commit — backend FPD prêt pour Render"
git branch -M main
git remote add origin https://github.com/<VOTRE_USER>/<VOTRE_REPO>.git
git push -u origin main
```

> ⚠️ Vérifiez que le dossier `models/` (avec les 8 fichiers `.pkl`, `.tflite`, `.json`) est bien commité — il pèse ~12 Mo, c'est OK pour GitHub.

### Étape 2 — Créer le service sur Render

#### Option A — Via Blueprint (recommandé, tout automatique)

1. Connectez-vous sur [dashboard.render.com](https://dashboard.render.com).
2. Cliquez sur **"New +"** → **"Blueprint"**.
3. Sélectionnez votre dépôt GitHub.
4. Render lit automatiquement `render.yaml` et vous montre le service à créer.
5. Cliquez sur **"Apply"** — le déploiement démarre.

#### Option B — Via Web Service manuel

1. **"New +"** → **"Web Service"**.
2. Connectez votre dépôt.
3. Renseignez :
   - **Environment** : `Python 3`
   - **Build Command** : `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --preload`
   - **Health Check Path** : `/health`
   - **Plan** : `Free` (ou supérieur)
4. Cliquez sur **"Create Web Service"**.

### Étape 3 — (Optionnel) Renseigner les variables d'environnement

Dans **Settings → Environment** du service :

| Clé | Valeur recommandée | Obligatoire |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://votre-frontend.com` (ou `*`) | Non |
| `LSTM_THRESHOLD` | `0.5` | Non |
| `HTTP_TIMEOUT` | `15` | Non |
| `GITHUB_TOKEN` | _votre token GitHub_ | Non, mais conseillé (60 → 5 000 req/h) |

> Le `render.yaml` fournit déjà des valeurs par défaut sensées. Vous n'avez besoin d'ajouter manuellement que `GITHUB_TOKEN` si vous souhaitez l'utiliser.

### Étape 4 — Vérifier le déploiement

Une fois "Live" (1ʳᵉ build : ~3-5 min) :

```bash
curl https://<votre-service>.onrender.com/health
curl https://<votre-service>.onrender.com/api/models
```

Vous devriez voir un JSON `{"status": "ok", "version": "...", ...}`.

---

## ⚠️ Limitations du Free Tier Render

- **512 Mo de RAM** : suffisant grâce à l'allègement TF → TFLite, mais juste. Si vous observez des OOM, passez au plan **Starter ($7/mois, 512 Mo + pas de cold-start)**.
- **Cold-start ~30 s** après 15 min d'inactivité : le service se met en veille. La première requête réveille les modèles.
- **Pas de stockage persistant** : OK ici car les modèles sont commités dans le dépôt.

---

## 🧪 Tester en local avant de déployer

```bash
# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

# Installer les dépendances
pip install -r requirements.txt

# Lancer en mode production (identique à Render)
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 120 --preload

# Ou en mode dev
python app.py
```

Tester :
```bash
curl http://localhost:5000/health
```

---

## 🐛 Dépannage rapide

| Symptôme | Cause probable | Solution |
|---|---|---|
| `ModuleNotFoundError: tflite_runtime` | Mauvaise version de Python | Vérifier que `runtime.txt` contient `python-3.11.9` (tflite-runtime 2.14 supporte 3.9-3.11) |
| Build > 500 Mo | tensorflow réintroduit accidentellement | Vérifier que `tensorflow` est ABSENT de `requirements.txt` |
| `502 Bad Gateway` au démarrage | Modèles trop lourds → OOM | Passer au plan Starter, ou retirer les modèles non utilisés |
| Cold start très lent | Free tier en veille | Ajouter un cron externe (UptimeRobot, cron-job.org) qui ping `/health` toutes les 10 min |
| CORS bloqué côté front | `ALLOWED_ORIGINS=*` mais navigateur strict | Mettre la valeur exacte du domaine du front |

---

## 📞 Endpoints disponibles

| Méthode | URL | Description |
|---|---|---|
| `GET` | `/` | Infos API |
| `GET` | `/health` | Healthcheck (utilisé par Render) |
| `GET` | `/api/models` | Liste des modèles ML chargés |
| `POST` | `/api/analyze` | Analyse manuelle d'un profil |
| `POST` | `/api/analyze-job` | Analyse LSTM d'une offre d'emploi |
| `POST` | `/api/extract-url` | Extraction brute d'un profil depuis URL |
| `POST` | `/api/analyze-url` | Extraction + scoring complet depuis URL |

Bon déploiement ! 🎉
