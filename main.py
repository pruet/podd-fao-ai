import os
import io
import hmac
import hashlib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
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
import uuid

load_dotenv()

from google.cloud import datastore

IS_GAE = os.getenv("GAE_ENV") == "standard" or "GAE_SERVICE" in os.environ
if IS_GAE:
    DB_PATH = "/tmp/api_logs.db"
    LOGS_DIR = "/tmp/logs"
else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_logs.db")
    LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

USE_DATASTORE = IS_GAE or os.getenv("USE_DATASTORE") == "true"
datastore_client = None

if USE_DATASTORE:
    try:
        datastore_client = datastore.Client()
    except Exception as e:
        print(f"Failed to initialize Datastore client, falling back to SQLite: {e}")
        USE_DATASTORE = False


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
                response_body TEXT,
                steps_json TEXT,
                image_path TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0
            )
        """)
        # Run migrations if columns are missing
        cursor.execute("PRAGMA table_info(api_logs)")
        columns = [row[1] for row in cursor.fetchall()]
        if "image_path" not in columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN image_path TEXT")
        if "prompt_tokens" not in columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN prompt_tokens INTEGER DEFAULT 0")
        if "output_tokens" not in columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN output_tokens INTEGER DEFAULT 0")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing DB: {e}")

# Initialize DB on load
init_db()

security = HTTPBasic()

def authenticate_dashboard(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("DASHBOARD_USERNAME", "admin")
    correct_password = os.getenv("DASHBOARD_PASSWORD", "PoddFaoSecure2026!")
    
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

def log_request_response(method: str, path: str, status_code: int, latency: float, client_ip: str, request_params: dict, response_body: str, steps_json: Optional[dict] = None, image_path: Optional[str] = None, prompt_tokens: int = 0, output_tokens: int = 0):
    def make_serializable(item):
        if isinstance(item, dict):
            return {k: make_serializable(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [make_serializable(i) for i in item]
        elif isinstance(item, (str, int, float, bool, type(None))):
            return item
        else:
            return str(item)

    if USE_DATASTORE and datastore_client:
        try:
            key = datastore_client.key("ApiLog")
            entity = datastore.Entity(
                key=key,
                exclude_from_indexes=("response_body", "steps_json", "request_params", "image_path")
            )
            entity.update({
                "timestamp": datetime.utcnow().isoformat(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency": latency,
                "client_ip": client_ip,
                "request_params": json.dumps(make_serializable(request_params)),
                "response_body": response_body,
                "steps_json": json.dumps(make_serializable(steps_json)) if steps_json else None,
                "image_path": image_path,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens
            })
            datastore_client.put(entity)
            return
        except Exception as e:
            print(f"Error logging to Datastore: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_logs (timestamp, method, path, status_code, latency, client_ip, request_params, response_body, steps_json, image_path, prompt_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            method,
            path,
            status_code,
            latency,
            client_ip,
            json.dumps(make_serializable(request_params)),
            response_body,
            json.dumps(make_serializable(steps_json)) if steps_json else None,
            image_path,
            prompt_tokens,
            output_tokens
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to DB: {e}")


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "guardrail_policy": "Determine if the image contains an animal or animal body part.",
        "prompt_template": "Analyze the provided image(s) and description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal.\n2. List up to three possible diseases affecting the animal in the image(s).\n3. For each disease, provide a confidence level (0.0 to 1.0) and brief reasoning/symptoms observed.\n\nConstraints:\n- You MUST output the entire response (including animal type, disease names, reasoning, and invalid_reason) in {lang_name} language.\n- If a text description is provided below, incorporate it into your analysis:\n  Description: {description}"
    }

app = FastAPI(
    title="FAO-PODD Animal Disease Diagnosis API",
    description="Analyze animal images and descriptions to identify potential diseases.",
    version="2.4.1"
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

async def fetch_lahis_report_images(report_id: str, token: str, api_url: str) -> List[bytes]:
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
            
        images_bytes = []
        for img in images:
            content_path = None
            if isinstance(img, dict):
                content_path = img.get("links", {}).get("content")
            if content_path:
                download_url = f"{api_url.rstrip('/')}{content_path}"
                try:
                    img_response = await client.get(download_url, headers=headers, timeout=20.0)
                    img_response.raise_for_status()
                    images_bytes.append(img_response.content)
                except Exception as e:
                    print(f"Failed to download image from {download_url}: {e}")
                    
        if not images_bytes:
            raise HTTPException(
                status_code=404,
                detail=f"No downloadable images found in LAHIS report {report_id}."
            )
        return images_bytes

def verify_webhook_signature(path: str, timestamp: str, raw_body: bytes, signature: str) -> bool:
    signing_secret = os.getenv("LAHIS_SIGNING_SECRET")
    if not signing_secret:
        return True
    message = f"POST\n{path}\n{timestamp}\n".encode("utf-8") + raw_body
    expected_signature = hmac.new(
        signing_secret.encode("utf-8"),
        message,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_signature, signature)

def format_diagnosis_comment(result: dict, lang: str) -> str:
    animal_type = result.get("animal_type") or "Unknown"
    is_valid = result.get("is_valid_animal_image")
    
    if not is_valid:
        reason = result.get("invalid_reason") or "Invalid image"
        if lang == "lo":
            return f"ຜົນການວິເຄາະ AI: ຮູບພາບບໍ່ຖືກຕ້ອງ. ເຫດຜົນ: {reason}"
        elif lang == "th":
            return f"ผลการวิเคราะห์ AI: รูปภาพไม่ถูกต้อง เหตุผล: {reason}"
        return f"AI Analysis Result: Invalid image. Reason: {reason}"
        
    diseases_list = []
    for d in (result.get("diseases") or []):
        name = d.get("name")
        conf = d.get("confidence", 0.0) * 100
        reason = d.get("reasoning") or ""
        diseases_list.append(f"- {name} ({conf:.0f}%): {reason}")
        
    diseases_str = "\n".join(diseases_list)
    if lang == "lo":
        return f"ຜົນການວິເຄາະ AI:\n- ປະເພດສັດ: {animal_type}\n- ພະຍາດທີ່ເປັນໄປໄດ້:\n{diseases_str}"
    elif lang == "th":
        return f"ผลการวิเคราะห์ AI:\n- ประเภทสัตว์: {animal_type}\n- โรคที่เป็นไปได้:\n{diseases_str}"
    return f"AI Analysis Result:\n- Animal Type: {animal_type}\n- Potential Diseases:\n{diseases_str}"

async def submit_lahis_comment(report_id: str, event_id: str, body_text: str, confidence: float, tenant_api_url: str, token: str):
    async with httpx.AsyncClient() as client:
        comments_url = f"{tenant_api_url.rstrip('/')}/api/integrations/v1/reports/{report_id}/comments"
        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": f"ai-feedback-{event_id}",
            "Content-Type": "application/json"
        }
        payload = {
            "externalActionId": f"ai-feedback-{event_id}",
            "body": body_text,
            "visibility": "staff",
            "metadata": {
                "model": "gemini-3.5-flash",
                "confidence": confidence
            },
            "recommendation": {
                "type": "officer_review",
                "priority": "high" if confidence > 0.7 else "medium"
            }
        }
        print(f"[STEP 4] Request back to LAHIS. URL: {comments_url}, Payload: {json.dumps(payload)}")
        try:
            response = await client.post(comments_url, json=payload, headers=headers, timeout=15.0)
            print(f"[STEP 5] Response back from LAHIS. Status: {response.status_code}, Body: {response.text}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"[STEP 5 ERROR] LAHIS comments API returned error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"LAHIS comments API returned error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            print(f"[STEP 5 ERROR] Failed to submit comment to LAHIS: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to submit comment to LAHIS: {str(e)}"
            )

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_animal_image(
    request: Request,
    images: Optional[List[UploadFile]] = File(None, description="Images of the sick/diseased animal"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID to fetch the image from"),
    description: Optional[str] = Form(None, description="Optional text description of signs/symptoms"),
    lang: str = Form("en", description="Language for response: 'en' (English), 'th' (Thai), 'lo' (Lao)")
):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    saved_image_path = None
    prompt_tokens = 0
    output_tokens = 0
    
    # Initialize request_params early to prevent UnboundLocalError in exception handler
    request_params = {
        "lang": lang,
        "description": description,
        "report_id": report_id,
        "image_filenames": [img.filename for img in images] if images else [],
        "source": request.headers.get("x-source", "api")
    }
    steps = {
        "step_1": {
            "title": "1. Request from LAHIS",
            "data": request_params
        }
    }
    
    try:
        content_type = request.headers.get("content-type", "")
        is_json = "application/json" in content_type
        event_id = "unknown-event"
        
        if is_json:
            raw_body = await request.body()
            try:
                body = json.loads(raw_body)
                request_params = body
            except Exception:
                request_params = {"raw_body": raw_body.decode("utf-8", errors="ignore")}
                raise HTTPException(status_code=400, detail="Invalid JSON body")
                
            timestamp = request.headers.get("x-ohtk-timestamp", "")
            signature = request.headers.get("x-ohtk-signature", "")
            
            signing_secret = os.getenv("LAHIS_SIGNING_SECRET")
            if signing_secret:
                if not signature or not timestamp:
                    raise HTTPException(status_code=401, detail="Missing signature headers")
                
                path = request.url.path
                alt_path = path + "/" if not path.endswith("/") else path[:-1]
                
                sig_ok = verify_webhook_signature(path, timestamp, raw_body, signature)
                if not sig_ok:
                    sig_ok = verify_webhook_signature(alt_path, timestamp, raw_body, signature)
                    
                if not sig_ok:
                    raise HTTPException(status_code=401, detail="Invalid webhook signature")
                    
            event_type = body.get("eventType")
            if event_type != "report.submitted":
                raise HTTPException(status_code=400, detail=f"Unsupported eventType: {event_type}")
                
            report_data = body.get("report") or {}
            report_id = report_data.get("id")
            if not report_id:
                raise HTTPException(status_code=400, detail="Missing report.id in payload")
                
            event_id = body.get("eventId") or "unknown-event"
            lang = os.getenv("LAHIS_LANG", "lo")
            description = f"Webhook Event ID: {event_id}. Report Type: {report_data.get('reportType', {}).get('name', 'Unknown')}"
        else:
            request_params = {
                "lang": lang,
                "description": description,
                "report_id": report_id,
                "image_filenames": [img.filename for img in images] if images else [],
                "source": request.headers.get("x-source", "api")
            }
        steps["step_1"]["data"] = request_params
        print(f"[STEP 1] Request from LAHIS. Content-Type: {content_type}, is_json: {is_json}, params/body: {json.dumps(request_params)}")
        if lang not in ["en", "th", "lo"]:
            raise HTTPException(status_code=400, detail="Unsupported language. Supported languages are 'en', 'th', 'lo'.")
            
        if not is_json and not images and not report_id:
            raise HTTPException(status_code=400, detail="Either image files or report_id must be provided.")
        
        multiple_images_bytes = []
        api_url = os.getenv("TENANT_API_URL")
        client_id = os.getenv("LAHIS_CLIENT_ID")
        client_secret = os.getenv("LAHIS_CLIENT_SECRET")
        
        if images and not is_json:
            for img in images:
                try:
                    data = await img.read()
                    if data:
                        multiple_images_bytes.append(data)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid image file {img.filename}: {str(e)}")
        else:
            # Fetch image bytes from LAHIS API
            if not all([api_url, client_id, client_secret]):
                raise HTTPException(
                    status_code=500,
                    detail="LAHIS integration environment variables (TENANT_API_URL, LAHIS_CLIENT_ID, LAHIS_CLIENT_SECRET) are not fully configured."
                )
                
            token = await get_lahis_token(client_id, client_secret, api_url)
            multiple_images_bytes = await fetch_lahis_report_images(report_id, token, api_url)
            
        pil_images = []
        for i, img_bytes in enumerate(multiple_images_bytes):
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                pil_images.append(pil_img)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse image index {i}: {str(e)}")
        
        # Save images to logs/images/
        saved_paths = []
        for i, img_bytes in enumerate(multiple_images_bytes):
            try:
                images_dir = os.path.join(LOGS_DIR, "images")
                os.makedirs(images_dir, exist_ok=True)
                ext = ".jpg"
                pil_img = pil_images[i]
                if pil_img.format:
                    ext = f".{pil_img.format.lower()}"
                saved_image_filename = f"{uuid.uuid4()}{ext}"
                target_path = os.path.join(images_dir, saved_image_filename)
                with open(target_path, "wb") as f:
                    f.write(img_bytes)
                saved_paths.append(f"logs/images/{saved_image_filename}")
            except Exception as e:
                print(f"Error saving image {i} to logs: {e}")
        if saved_paths:
            saved_image_path = ",".join(saved_paths)
        
        # Get initialized Gemini Client
        genai_client = get_genai_client()
        
        # Define prompt based on language
        lang_names = {"en": "English", "th": "Thai", "lo": "Lao"}
        lang_name = lang_names[lang]
        
        # Load guardrail policy and prompt_template from config.json
        config = load_config()
        guardrail_policy = config.get("guardrail_policy", "")
        prompt_template = config.get(
            "prompt_template",
            "Analyze the provided image(s) and description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal.\n2. List up to three possible diseases affecting the animal in the image(s).\n3. For each disease, provide a confidence level (0.0 to 1.0) and brief reasoning/symptoms observed.\n\nConstraints:\n- You MUST output the entire response (including animal type, disease names, reasoning, and invalid_reason) in {lang_name} language.\n- If a text description is provided below, incorporate it into your analysis:\n  Description: {description}"
        )
        
        try:
            prompt = prompt_template.format(
                guardrail_policy=guardrail_policy,
                lang_name=lang_name,
                description=description or 'None provided'
            )
        except Exception as e:
            print(f"Error formatting prompt_template from config.json: {e}")
            prompt = f"Analyze the provided image(s) and description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal.\n2. List up to three possible diseases affecting the animal in the image(s).\n3. For each disease, provide a confidence level (0.0 to 1.0) and brief reasoning/symptoms observed.\n\nConstraints:\n- You MUST output the entire response (including animal type, disease names, reasoning, and invalid_reason) in {lang_name} language.\n- If a text description is provided below, incorporate it into your analysis:\n  Description: {description or 'None provided'}"
        
        steps["step_2"] = {
            "title": "2. Request to Google AI",
            "data": {
                "model": "gemini-3.5-flash",
                "prompt": prompt
            }
        }
        print(f"[STEP 2] Request to Google AI. Model: gemini-3.5-flash, Prompt length: {len(prompt)}")
        # Run inference using gemini-3.5-flash
        response = genai_client.models.generate_content(
            model='gemini-3.5-flash',
            contents=pil_images + [prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiagnosisResponse,
                temperature=0.2,
            ),
        )
        print(f"[STEP 3] Response from Google AI: {response.text}")
        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        result = json.loads(response.text)
        steps["step_3"] = {
            "title": "3. Response from Google AI",
            "data": {
                "result": result,
                "usage_metadata": {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens
                }
            }
        }
        
        # If it was a webhook, post the feedback comment back to LAHIS
        if is_json:
            comment_body = format_diagnosis_comment(result, lang)
            # Find highest confidence score from diseases
            highest_confidence = 0.0
            for d in (result.get("diseases") or []):
                highest_confidence = max(highest_confidence, d.get("confidence", 0.0))
            
            # Re-fetch or reuse token to call comments API
            token = await get_lahis_token(client_id, client_secret, api_url)
            
            comments_url = f"{api_url.rstrip('/')}/api/integrations/v1/reports/{report_id}/comments"
            payload = {
                "externalActionId": f"ai-feedback-{event_id}",
                "body": comment_body,
                "visibility": "staff",
                "metadata": {
                    "model": "gemini-3.5-flash",
                    "confidence": highest_confidence
                },
                "recommendation": {
                    "type": "officer_review",
                    "priority": "high" if highest_confidence > 0.7 else "medium"
                }
            }
            steps["step_4"] = {
                "title": "4. Request back to LAHIS",
                "data": {
                    "url": comments_url,
                    "payload": payload
                }
            }
            try:
                comment_response = await submit_lahis_comment(report_id, event_id, comment_body, highest_confidence, api_url, token)
                steps["step_5"] = {
                    "title": "5. Response back from LAHIS",
                    "data": comment_response
                }
            except HTTPException as e:
                steps["step_5"] = {
                    "title": "5. Response back from LAHIS (Error)",
                    "data": {
                        "status_code": e.status_code,
                        "detail": e.detail
                    }
                }
                # Log intermediate steps before raising
                latency = time.time() - start_time
                log_request_response(
                    method="POST",
                    path="/analyze",
                    status_code=e.status_code,
                    latency=latency,
                    client_ip=client_ip,
                    request_params=request_params,
                    response_body=json.dumps({"detail": e.detail}),
                    steps_json=steps,
                    image_path=saved_image_path,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens
                )
                raise e
        
        # Log success
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/analyze",
            status_code=200,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps(result),
            steps_json=steps,
            image_path=saved_image_path,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
        )
        return result

    except HTTPException as e:
        # Avoid double logging if already logged in the inner try block
        # (Though status_code could be 502, let's log if not already written)
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/analyze",
            status_code=e.status_code,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": e.detail}),
            steps_json=steps,
            image_path=saved_image_path,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
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
            response_body=json.dumps({"detail": str(e)}),
            steps_json=steps,
            image_path=saved_image_path,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens
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
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ApiLog")
            query.order = ["-timestamp"]
            results = list(query.fetch(limit=limit))
            
            logs = []
            for entity in results:
                raw_params = entity.get("request_params")
                if isinstance(raw_params, str):
                    try:
                        parsed_params = json.loads(raw_params)
                    except Exception:
                        parsed_params = {"raw": raw_params}
                elif isinstance(raw_params, dict):
                    parsed_params = raw_params
                else:
                    parsed_params = {}

                raw_steps = entity.get("steps_json")
                if isinstance(raw_steps, str):
                    try:
                        parsed_steps = json.loads(raw_steps)
                    except Exception:
                        parsed_steps = None
                elif isinstance(raw_steps, dict):
                    parsed_steps = raw_steps
                else:
                    parsed_steps = None

                logs.append({
                    "id": entity.key.id,
                    "timestamp": entity.get("timestamp"),
                    "method": entity.get("method"),
                    "path": entity.get("path"),
                    "status_code": entity.get("status_code"),
                    "latency": entity.get("latency") or 0.0,
                    "client_ip": entity.get("client_ip"),
                    "request_params": parsed_params,
                    "response_body": entity.get("response_body"),
                    "steps_json": parsed_steps,
                    "image_path": entity.get("image_path"),
                    "prompt_tokens": entity.get("prompt_tokens", 0),
                    "output_tokens": entity.get("output_tokens", 0)
                })
            return logs
        except Exception as e:
            print(f"Datastore fetch error: {e}")

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
                "response_body": row["response_body"],
                "steps_json": json.loads(row["steps_json"]) if ("steps_json" in row.keys() and row["steps_json"]) else None,
                "image_path": row["image_path"] if ("image_path" in row.keys() and row["image_path"]) else None,
                "prompt_tokens": row["prompt_tokens"] if "prompt_tokens" in row.keys() else 0,
                "output_tokens": row["output_tokens"] if "output_tokens" in row.keys() else 0
            })
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")

@app.get("/api/logs/image/{filename}")
async def get_log_image(filename: str, username: str = Depends(authenticate_dashboard)):
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    images_dir = os.path.join(LOGS_DIR, "images")
    file_path = os.path.join(images_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port)

