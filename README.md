# FAO-PODD Animal Disease Diagnosis API & Dashboard

A professional FastAPI-based service designed to assist in animal disease diagnostics by analyzing animal images and symptom descriptions using the Gemini 3.5 Flash model. It is designed to work standalone or integrate seamlessly with the **LAHIS** API system.

---

## 🌟 Features

- **Double Input Channel**:
  - **Direct Upload**: Send images directly via POST requests.
  - **LAHIS Integration**: Automatically fetches reports and downloads corresponding animal images from the LAHIS server using OAuth2 credentials.
- **AI Diagnostics**: Leverages Gemini 3.5 Flash to identify animal type, detect up to 3 potential diseases with confidence scores, and provide diagnostic reasoning.
- **Interactive Live Dashboard**: A modern, dark-mode glassmorphic web dashboard served at `/` to monitor API performance, track latency, and review recent request/response logs.
- **Secure Log Monitor**:
  - Automatically captures requests and response payloads to a local SQLite database (`/tmp/api_logs.db`).
  - Protected by HTTP Basic Authentication to prevent unauthorized access.
- **Production-Ready**: Designed for easy deployment to **Google App Engine (GAE)** standard environment.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **AI Inference**: Google GenAI SDK (Gemini 3.5 Flash)
- **Database**: SQLite3
- **Deployment**: Google App Engine (Standard)
- **Dashboard UI**: Modern Vanilla HTML5 / CSS3 / JavaScript (Outfit typography, glassmorphism, responsive grid)

---

## ⚙️ Configuration (`.env`)

Create a `.env` file in the source folder based on the provided `.env.example`:

```env
# Gemini API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Port and Host settings
PORT=8000
HOST=0.0.0.0

# LAHIS Integration Credentials
TENANT_API_URL=https://demo.api.lahis.ohtk.org
LAHIS_CLIENT_ID=your_client_id_here
LAHIS_CLIENT_SECRET=your_client_secret_here
LAHIS_SIGNING_SECRET=your_webhook_signing_secret_here
LAHIS_LANG=lo

# Web Dashboard Protection
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=admin
```

---

## 🚀 Getting Started

### 1. Installation

Set up a virtual environment and install the required dependencies:

```bash
# Clone the repository
git clone git@github.com:pruet/podd-fao-ai.git
cd podd-fao-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Locally

Start the development server:

```bash
uvicorn main:app --reload
```

- **API Swagger Docs**: Visit [http://localhost:8000/docs](http://localhost:8000/docs)
- **Log Dashboard**: Visit [http://localhost:8000/](http://localhost:8000/) (Login using dashboard credentials configured in `.env`).

### 3. Running Unit Tests

Execute the test suite using Python's built-in unittest framework:

```bash
python -m unittest test_main.py
```

---

## 📦 App Engine Deployment

Deploy the service to Google App Engine standard environment in a few commands:

```bash
# Set your active GCP account
gcloud config set account YOUR_ACCOUNT_EMAIL

# Set the active project
gcloud config set project podd-fao-ai

# Deploy the application
gcloud app deploy app.yaml
```

The App Engine instance will run standard Python 3.11 environment using variables specified under `env_variables` in `app.yaml`.
