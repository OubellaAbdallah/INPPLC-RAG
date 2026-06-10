# Assistant RAG de Consultation Juridique — INPPLC

Ce projet implémente un assistant intelligent basé sur la **Génération Augmentée de Récupération (RAG)** pour l'**Instance Nationale de la Probité, de la Prévention et de la Lutte contre la Corruption (INPPLC)** du Royaume du Maroc. Il est conçu pour orienter et répondre aux questions relatives aux lois anti-corruption, aux déclarations de patrimoine, aux rapports d'évaluation nationaux et à la bonne gouvernance.

---

## 🚀 Fonctionnalités Clés

L'assistant intègre des composants avancés pour offrir une expérience fiable, sécurisée et transparente :

### 1. Ingestion Multimodale de Documents (Mistral OCR + LLM)
- Analyse et extrait le contenu textuel de documents complexes (PDFs officiels, scans) via l'API Mistral OCR.
- Effectue un post-traitement par grand modèle linguistique (LLM) afin de traduire sous forme de descriptions textuelles naturelles les tableaux complexes, graphiques, figures et schémas tout en conservant l'exactitude des données.
- Utilise une empreinte SHA-256 (manifeste d'idempotence) pour éviter de traiter deux fois les mêmes documents.

### 2. Guardrails et Classification Automatique des Requêtes
- Analyse et classifie chaque message utilisateur en trois types de requêtes :
  - **RAG_REQUIRED :** Requête factuelle relative au domaine de l'INPPLC nécessitant une recherche documentaire.
  - **DIRECT_RESPONSE :** Salutations ou questions d'ordre général ne nécessitant pas de recherche documentaire.
  - **OUT_OF_SCOPE :** Sujet hors du cadre réglementaire/corruption. Le système refuse poliment de répondre pour éviter les hallucinations hors domaine.

### 3. Réécriture de Requête Basée sur l'Historique (Query Rewriting)
- Reformule la requête de l'utilisateur en intégrant le contexte de la conversation en cours afin de maintenir une continuité parfaite lors d'une discussion en plusieurs étapes.

### 4. Recherche Hybride Sémantique et Lexicale (Dense & Sparse)
- Combine la recherche sémantique dense basée sur les plongements vectoriels (embeddings `BAAI/bge-large-en-v1.5` stockés sous ChromaDB) et la recherche lexicale sparse (TF-IDF & BM25) pour garantir un taux de rappel et de précision maximal.

### 5. Streaming Asynchrone en Temps Réel (Server-Sent Events)
- L'API FastAPI diffuse la réponse générée par Mistral Large mot par mot (streaming token-by-token via SSE) pour minimiser le temps d'attente utilisateur.

### 6. Traçabilité des Sources et Panneau d'Audit (Audit Trail)
- Transparence totale : l'interface web comprend un volet latéral d'audit affichant pour chaque réponse de l'assistant les extraits exacts des documents consultés, leur score de pertinence et leur fichier source.

### 7. Gestion de Sessions et Historique Persistant
- Stocke l'historique des conversations dans une base de données JSON locale (`data/chat_sessions.json`), permettant de reprendre n'importe quelle session de chat précédente.

### 8. Exportation de Notes Juridiques au Format PDF
- Génère instantanément une note de consultation structurée (Faits, Analyse, Recommandations, Sources juridiques et scores de pertinence) téléchargeable directement au format PDF grâce à `html2pdf.js`.

---

## 🛠️ Architecture Technique

```
                       Message de l'utilisateur
                                  │
                                  ▼
             Classification & Guardrail (Outils LLM)
             ├── DIRECT_RESPONSE -> Réponse immédiate
             ├── OUT_OF_SCOPE    -> Refus poli
             └── RAG_REQUIRED    -> Poursuite du pipeline
                                  │
                                  ▼
                 Réécriture de la Requête (Mistral)
               (Prend en compte l'historique de chat)
                                  │
                                  ▼
                      Recherche Hybride (Dense + Sparse)
            ├── Dense : ChromaDB (Embedding: BGE-large-en-v1.5)
            └── Sparse : TF-IDF / BM25
                                  │
                                  ▼
                     Génération Stream (SSE)
                       (Mistral Large API)
                                  │
                                  ▼
                Rapport Web + Panneau d'Audit + PDF
```

---

## 📂 Structure du Projet

```
Inpplc-RAG/
├── api.py                      # Serveur principal FastAPI (SSE, endpoints de chat et sessions)
├── requirements.txt            # Liste des dépendances Python requises pour la production
├── pyproject.toml              # Fichier de configuration Poetry pour l'environnement dev
├── webapp/                     # Partie client (Interface Web)
│   ├── index.html              # Structure de l'application (Single Page App)
│   ├── style.css               # Feuille de style (Thème moderne vert #1B5E35 et or #C8A84B)
│   ├── app.js                  # Logique d'affichage, streaming SSE, export PDF et sessions
│   └── assets/                 # Ressources statiques (logo INPPLC)
├── src/                        # Code source Python interne
│   ├── config/                 # Gestion des configurations par Pydantic Settings
│   ├── ingestion/              # Extraction, captioning et chunking de documents
│   ├── retrieval/              # Indexation dense, sparse, recherche hybride et re-ranking
│   ├── generation/             # Gestion des invites (prompts), classification et appel LLM
│   └── utils/                  # Outils d'aide (système de logs structurés, formateurs)
├── data/                       # Conteneur des données de l'application
│   ├── raw/                    # Dossier source des documents officiels (PDF, TXT)
│   ├── processed/              # Fichiers de cache de parsing intermédiaire
│   └── indices/                # Fichiers d'indexation vectorielle locale (ChromaDB, BM25)
└── scripts/                    # Scripts utilitaires d'administration et d'évaluation
    ├── run_ingestion.py        # Démarre l'analyse et l'indexation des documents bruts
    ├── cleanup_junk_chunks.py  # Supprime les chunks de texte invalides de la base vectorielle
    ├── test_rag_terminal.py    # Client CLI pour tester le pipeline RAG dans le terminal
    └── run_evaluation.py       # Exécute l'évaluation automatisée Ragas
```

---

## ⚙️ Configuration & Installation

### Prérequis
- **Python 3.11** ou supérieur
- Une clé API Mistral valide

### 1. Installation des Dépendances
Dans la racine du projet, créez un environnement virtuel et installez les dépendances :
```powershell
# Création de l'environnement virtuel
python -m venv .venv

# Activation (Windows)
.venv\Scripts\activate

# Activation (macOS/Linux)
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

### 2. Fichier d'Environnement
Copiez le modèle `.env.example` en un fichier `.env` et mettez-y vos clés API :
```bash
cp .env.example .env
```
Assurez-vous de configurer au moins la clé suivante :
```env
MISTRAL_API_KEY="votre_cle_api_mistral"
```

---

## 🚀 Comment Lancer l'Application

### Étape 1 : Indexer les Documents (Ingestion)
Avant de lancer le chatbot, vous devez parser et indexer les documents réglementaires situés dans `data/raw/` :
```bash
python -m scripts.run_ingestion
```
*Note : Si vous souhaitez tester rapidement sans générer les descriptions de tableaux/figures, utilisez l'option `--skip-captioning` :*
```bash
python -m scripts.run_ingestion --skip-captioning
```

### Étape 2 : Lancer le Serveur API
Démarrez le serveur FastAPI :
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
Le serveur sera en ligne sur `http://localhost:8000`.

### Étape 3 : Accéder à l'Interface Utilisateur
- **Accès Direct (Hébergement API) :** Ouvrez simplement [http://localhost:8000/app](http://localhost:8000/app) dans votre navigateur. Le serveur FastAPI sert directement le frontend.
- **Accès Client Autonome (Optionnel) :** Vous pouvez également servir l'interface via un simple serveur web local depuis le dossier webapp :
  ```bash
  cd webapp
  python -m http.server 3000
  ```
  Accédez ensuite à l'application sur [http://localhost:3000](http://localhost:3000).

---

## 🐳 Déploiement avec Docker

Le projet supporte un déploiement conteneurisé complet via Docker et Docker Compose.

### 1. Lancer l'application avec Docker Compose
Assurez-vous d'avoir configuré votre fichier `.env` avec votre clé Mistral. Lancez ensuite les conteneurs en tâche de fond :
```bash
docker-compose up --build -d
```
L'assistant sera accessible à l'adresse suivante : [http://localhost:8000/app](http://localhost:8000/app).

### 2. Initialisation et Ingestion (Optionnel)
Si la base de données vectorielle existe déjà sur votre machine hôte sous le dossier `data/indices/`, le conteneur l'utilisera directement via le montage de volume. L'assistant est donc immédiatement opérationnel.

Si vous souhaitez par la suite lancer ou mettre à jour l'ingestion de nouveaux documents placés dans `data/raw/` :
```bash
docker exec -it inpplc-chatbot python -m scripts.run_ingestion
```

---

## 🛠️ Commandes Utiles pour les Développeurs

- **Lancer les tests unitaires et d'intégration :**
  ```bash
  python -m pytest
  ```
- **Tester le RAG dans le terminal en interactif :**
  ```bash
  python -m scripts.test_rag_terminal
  ```
- **Nettoyer les données résiduelles ou trop courtes de la base vectorielle ChromaDB :**
  ```bash
  python -m scripts.cleanup_junk_chunks
  ```
- **Reconstruire les index de recherche sparse uniquement :**
  ```bash
  python -m scripts.rebuild_tfidf
  ```
- **Reconstruire uniquement les index sémantiques (dense) :**
  ```bash
  python -m scripts.rebuild_embeddings
  ```
