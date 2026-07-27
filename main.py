import os
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from PIL import Image
import json
import httpx
from dotenv import load_dotenv
import sqlite3
import time
from datetime import datetime

load_dotenv()

DB_PATH = "/tmp/api_logs.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                method TEXT,
                path TEXT,
                status_code INTEGER,
                latency REAL,
                client_ip TEXT,
                request_params TEXT,
                response_body TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing DB: {e}")

# Initialize DB on load
init_db()

security = HTTPBasic()

def authenticate_dashboard(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("DASHBOARD_USERNAME", "admin")
    correct_password = os.getenv("DASHBOARD_PASSWORD", "admin")
    
    import secrets
    is_correct_username = secrets.compare_digest(credentials.username, correct_username)
    is_correct_password = secrets.compare_digest(credentials.password, correct_password)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def log_request_response(method: str, path: str, status_code: int, latency: float, client_ip: str, request_params: dict, response_body: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_logs (timestamp, method, path, status_code, latency, client_ip, request_params, response_body)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            method,
            path,
            status_code,
            latency,
            client_ip,
            json.dumps(request_params),
            response_body
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to DB: {e}")

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(open_config := config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "guardrail_policy": "Determine if the image contains an animal or animal body part."
    }

app = FastAPI(
    title="FAO-PODD Animal Disease Diagnosis API",
    description="Analyze animal images and descriptions to identify potential diseases.",
    version="2.0"
)

# Initialize GenAI Client
# The client automatically picks up the GEMINI_API_KEY environment variable.
client = None

def get_genai_client():
    global client
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set.")
        client = genai.Client(api_key=api_key)
    return client

class DiseaseResponse(BaseModel):
    name: str = Field(description="Name of the disease in the requested language")
    confidence: float = Field(description="Confidence level of this disease, between 0.0 and 1.0")
    reasoning: str = Field(description="Brief clinical signs or reasoning supporting this diagnosis in the requested language")

class DiagnosisResponse(BaseModel):
    is_valid_animal_image: bool = Field(description="True if the image contains an animal or animal body parts, False otherwise")
    invalid_reason: Optional[str] = Field(None, description="Reason why the image is invalid in the requested language (if is_valid_animal_image is False)")
    animal_type: Optional[str] = Field(None, description="Type of animal identified in the requested language (null if invalid)")
    diseases: Optional[List[DiseaseResponse]] = Field(None, description="List of up to three potential diseases (null or empty if invalid)")

async def get_lahis_token(client_id: str, client_secret: str, api_url: str) -> str:
    async with httpx.AsyncClient() as client:
        token_url = f"{api_url.rstrip('/')}/o/token/"
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        try:
            response = await client.post(token_url, data=data, timeout=10.0)
            response.raise_for_status()
            res_json = response.json()
            return res_json["access_token"]
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to authenticate with LAHIS server: {str(e)}"
            )

async def fetch_lahis_report_image(report_id: str, token: str, api_url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        report_url = f"{api_url.rstrip('/')}/api/integrations/v1/reports/{report_id}/images"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = await client.get(report_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            report_data = response.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch report {report_id} images from LAHIS: {str(e)}"
            )
        
        images = report_data.get("images") or []
        if not images or not isinstance(images, list):
            raise HTTPException(
                status_code=404,
                detail=f"No animal image found in LAHIS report {report_id}."
            )
            
        first_image = images[0]
        content_path = None
        if isinstance(first_image, dict):
            content_path = first_image.get("links", {}).get("content")
            
        if not content_path:
            raise HTTPException(
                status_code=404,
                detail=f"No image download path found in LAHIS report {report_id}."
            )
            
        download_url = f"{api_url.rstrip('/')}{content_path}"
        try:
            img_response = await client.get(download_url, headers=headers, timeout=20.0)
            img_response.raise_for_status()
            return img_response.content
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download image from {download_url}: {str(e)}"
            )

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_animal_image(
    request: Request,
    image: Optional[UploadFile] = File(None, description="Image of the sick/diseased animal"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID to fetch the image from"),
    description: Optional[str] = Form(None, description="Optional text description of signs/symptoms"),
    lang: str = Form("en", description="Language for response: 'en' (English), 'th' (Thai), 'lo' (Lao)")
):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    request_params = {
        "lang": lang,
        "description": description,
        "report_id": report_id,
        "image_filename": image.filename if image else None
    }
    
    try:
        if lang not in ["en", "th", "lo"]:
            raise HTTPException(status_code=400, detail="Unsupported language. Supported languages are 'en', 'th', 'lo'.")
            
        if not image and not report_id:
            raise HTTPException(status_code=400, detail="Either image file or report_id must be provided.")
        
        image_bytes = None
        if image:
            # Read image bytes
            try:
                image_bytes = await image.read()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        elif report_id:
            # Fetch image bytes from LAHIS API
            api_url = os.getenv("TENANT_API_URL")
            client_id = os.getenv("LAHIS_CLIENT_ID")
            client_secret = os.getenv("LAHIS_CLIENT_SECRET")
            
            if not all([api_url, client_id, client_secret]):
                raise HTTPException(
                    status_code=500,
                    detail="LAHIS integration environment variables (TENANT_API_URL, LAHIS_CLIENT_ID, LAHIS_CLIENT_SECRET) are not fully configured."
                )
                
            token = await get_lahis_token(client_id, client_secret, api_url)
            image_bytes = await fetch_lahis_report_image(report_id, token, api_url)
            
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse retrieved image: {str(e)}")
        
        # Get initialized Gemini Client
        genai_client = get_genai_client()
        
        # Define prompt based on language
        lang_names = {"en": "English", "th": "Thai", "lo": "Lao"}
        lang_name = lang_names[lang]
        
        # Load guardrail policy from config.json
        config = load_config()
        guardrail_policy = config.get("guardrail_policy", "")
        
        prompt = f"""
        Analyze the provided image and description.
        
        Guardrail Policy:
        {guardrail_policy}
    
        Tasks (Execute ONLY if `is_valid_animal_image` is True):
        1. Identify the type of animal.
        2. List up to three possible diseases affecting the animal in the image.
        3. For each disease, provide a confidence level (0.0 to 1.0) and brief reasoning/symptoms observed.
        
        Constraints:
        - You MUST output the entire response (including animal type, disease names, reasoning, and invalid_reason) in {lang_name} language.
        - If a text description is provided below, incorporate it into your analysis:
          Description: {description or 'None provided'}
        """
        
        # Run inference using gemini-3.5-flash
        response = genai_client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiagnosisResponse,
                temperature=0.2,
            ),
        )
        result = json.loads(response.text)
        
        # Log success
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/analyze",
            status_code=200,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps(result)
        )
        return result

    except HTTPException as e:
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/analyze",
            status_code=e.status_code,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": e.detail})
        )
        raise e
    except Exception as e:
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/analyze",
            status_code=500,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": str(e)})
        )
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(username: str = Depends(authenticate_dashboard)):
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>FAO-PODD Dashboard</h1><p>dashboard.html not found</p>"

@app.get("/api/logs")
async def get_logs(limit: int = 50, username: str = Depends(authenticate_dashboard)):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM api_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "method": row["method"],
                "path": row["path"],
                "status_code": row["status_code"],
                "latency": row["latency"],
                "client_ip": row["client_ip"],
                "request_params": json.loads(row["request_params"]),
                "response_body": row["response_body"]
            })
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port)

