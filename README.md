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
## 🔌 API Endpoints

### 1. Diagnosis Endpoint: `POST /analyze`

Analyzes animal disease from a directly uploaded image or fetched dynamically from a LAHIS report ID.

* **Request Type**: `multipart/form-data`
* **Parameters**:
  - `image` (File, Optional): The binary image of the sick animal (JPEG/PNG).
  - `report_id` (Text, Optional): The report identifier from the LAHIS system. If provided, the API will fetch the image dynamically from LAHIS.
  - `description` (Text, Optional): Symptoms, behaviors, or clinical signs observed.
  - `lang` (Text, Optional): The desired response language. Supported: `en` (English), `th` (Thai), `lo` (Lao). Defaults to `en`.

* **Response Example (`200 OK`)**:
  ```json
  {
    "is_valid_animal_image": true,
    "invalid_reason": null,
    "animal_type": "Swine",
    "diseases": [
      {
        "name": "African Swine Fever",
        "confidence": 0.95,
        "reasoning": "Observed high fever, dark spots on skin, and extreme lethargy consistent with clinical symptoms of ASF."
      }
    ]
  }
  ```

---

### 2. Live Dashboard: `GET /`

Served as HTML, protected by HTTP Basic Authentication.
* **Credentials**: Match the configured `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD`.

---

### 3. Fetch Request Logs: `GET /api/logs`

Fetches recent diagnostic requests and their responses. Protected by Basic Authentication.
* **Parameters**:
  - `limit` (Query, Optional): Maximum number of log records to return. Defaults to `50`.

* **Response Example (`200 OK`)**:
  ```json
  [
    {
      "id": 1,
      "timestamp": "2026-07-27T12:00:00.123456",
      "method": "POST",
      "path": "/analyze",
      "status_code": 200,
      "latency": 1.45,
      "client_ip": "127.0.0.1",
      "request_params": {
        "lang": "en",
        "description": "Limping cow",
        "report_id": null,
        "image_filename": "cow.jpg"
      },
      "response_body": "{\"is_valid_animal_image\":true,...}"
    }
  ]
  ```

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
