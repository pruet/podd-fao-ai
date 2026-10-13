import os
import io
import hmac
import hashlib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from PIL import Image
import json
import csv
import httpx
from dotenv import load_dotenv
import sqlite3
import time
from datetime import datetime, timezone
import uuid

load_dotenv()

from google.cloud import datastore
from google.cloud import storage

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

# Google Cloud Storage for permanent audit image storage
GCS_IMAGE_BUCKET = os.getenv("GCS_IMAGE_BUCKET", "podd-fao-ai.appspot.com" if (IS_GAE or os.getenv("GOOGLE_CLOUD_PROJECT") == "podd-fao-ai") else None)
storage_client = None

def get_storage_client():
    global storage_client
    if storage_client is None:
        try:
            storage_client = storage.Client()
        except Exception as e:
            print(f"Failed to initialize Storage client: {e}")
    return storage_client

def save_audit_image(img_bytes: bytes, filename: str, content_type: str = "image/jpeg"):
    # 1. Save locally for instant serving and fallback
    images_dir = os.path.join(LOGS_DIR, "images")
    try:
        os.makedirs(images_dir, exist_ok=True)
        target_path = os.path.join(images_dir, filename)
        with open(target_path, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        print(f"Error saving image locally: {e}")

    # 2. Upload to Google Cloud Storage for permanent audit storage
    bucket_name = GCS_IMAGE_BUCKET
    if bucket_name:
        try:
            client = get_storage_client()
            if client:
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"audit_images/{filename}")
                blob.upload_from_string(img_bytes, content_type=content_type)
                print(f"[GCS AUDIT] Saved gs://{bucket_name}/audit_images/{filename}")
        except Exception as e:
            print(f"[GCS AUDIT WARNING] Could not upload audit image to GCS: {e}")

def get_audit_image_bytes(filename: str) -> Optional[bytes]:
    # 1. Check local disk first
    images_dir = os.path.join(LOGS_DIR, "images")
    file_path = os.path.join(images_dir, filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception:
            pass

    # 2. Check permanent Google Cloud Storage bucket
    bucket_name = GCS_IMAGE_BUCKET
    if bucket_name:
        try:
            client = get_storage_client()
            if client:
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"audit_images/{filename}")
                if blob.exists():
                    img_bytes = blob.download_as_bytes()
                    # Cache locally for future requests
                    try:
                        os.makedirs(images_dir, exist_ok=True)
                        with open(file_path, "wb") as f:
                            f.write(img_bytes)
                    except Exception:
                        pass
                    return img_bytes
        except Exception as e:
            print(f"[GCS AUDIT WARNING] Could not download audit image from GCS: {e}")

    return None



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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                date_str TEXT,
                report_id TEXT,
                village_id TEXT,
                village_name TEXT,
                raw_payload TEXT
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

def record_cluster_report(report_id: str, village_id: str, village_name: Optional[str], date_str: str, raw_payload: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    if USE_DATASTORE and datastore_client:
        try:
            key = datastore_client.key("ClusterReport")
            entity = datastore.Entity(key=key, exclude_from_indexes=("raw_payload",))
            entity.update({
                "timestamp": timestamp,
                "date_str": str(date_str),
                "report_id": str(report_id),
                "village_id": str(village_id),
                "village_name": str(village_name or ""),
                "raw_payload": raw_payload
            })
            datastore_client.put(entity)
            return
        except Exception as e:
            print(f"Error logging ClusterReport to Datastore: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cluster_reports (timestamp, date_str, report_id, village_id, village_name, raw_payload)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, str(date_str), str(report_id), str(village_id), str(village_name or ""), raw_payload))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving cluster report to DB: {e}")

def get_village_reports_for_date(village_id: str, date_str: str) -> List[dict]:
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ClusterReport")
            query.add_filter("village_id", "=", str(village_id))
            query.add_filter("date_str", "=", str(date_str))
            results = list(query.fetch())
            return [
                {
                    "id": entity.key.id,
                    "timestamp": entity.get("timestamp"),
                    "date_str": entity.get("date_str"),
                    "report_id": entity.get("report_id"),
                    "village_id": entity.get("village_id"),
                    "village_name": entity.get("village_name"),
                    "raw_payload": entity.get("raw_payload")
                }
                for entity in results
            ]
        except Exception as e:
            print(f"Error querying ClusterReport from Datastore: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM cluster_reports WHERE village_id = ? AND date_str = ?",
            (str(village_id), str(date_str))
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "date_str": row["date_str"],
                "report_id": row["report_id"],
                "village_id": row["village_id"],
                "village_name": row["village_name"],
                "raw_payload": row["raw_payload"]
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Error querying cluster reports from DB: {e}")
        return []

def get_village_all_reports(village_id: str) -> List[dict]:
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ClusterReport")
            query.add_filter("village_id", "=", str(village_id))
            results = list(query.fetch())
            return [
                {
                    "id": entity.key.id,
                    "timestamp": entity.get("timestamp"),
                    "date_str": entity.get("date_str"),
                    "report_id": entity.get("report_id"),
                    "village_id": entity.get("village_id"),
                    "village_name": entity.get("village_name"),
                    "raw_payload": entity.get("raw_payload")
                }
                for entity in results
            ]
        except Exception as e:
            print(f"Error querying ClusterReport from Datastore: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM cluster_reports WHERE village_id = ?",
            (str(village_id),)
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "date_str": row["date_str"],
                "report_id": row["report_id"],
                "village_id": row["village_id"],
                "village_name": row["village_name"],
                "raw_payload": row["raw_payload"]
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Error querying cluster reports from DB: {e}")
        return []


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
            utc_now_iso = datetime.now(timezone.utc).isoformat()
            entity.update({
                "timestamp": utc_now_iso,
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
            datetime.now(timezone.utc).isoformat(),
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


def load_animals_and_diseases():
    diseases_path = os.path.join(os.path.dirname(__file__), "diseases.json")
    if os.path.exists(diseases_path):
        with open(diseases_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "Chicken": ["Avian influenza", "Newcastle disease", "Fowl Cholera"],
        "Goose and Duck": ["Avian influenza", "Newcastle disease", "Fowl Cholera", "Duck Plague"],
        "Pig": [
            "Classical Swine Fever",
            "African Swine Fever",
            "Porcine Reproductive and Respiratory Syndrome (PRRS)",
            "Leptospirosis",
            "Rabies",
            "Trichinellosis",
            "Foot and Mouth Disease (FMD)"
        ],
        "Cattle and Buffalo": [
            "Foot and Mouth Disease (FMD)",
            "Haemorrhagic Septicemia",
            "Anthrax",
            "Black Leg",
            "Lumpy Skin Disease",
            "Leptospirosis",
            "Rabies"
        ],
        "Sheep and Goat": [
            "Rabies",
            "Leptospirosis",
            "Foot and Mouth Disease (FMD)",
            "Peste des Petits Ruminants (PPR)"
        ],
        "Dog and Cat": ["Rabies"]
    }


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "guardrail_policy": "Determine if the image depicts a living animal, animal carcass, or animal body parts (such as skin, mouth, organs, legs, etc.) of an animal belonging to the ALLOWED animal species list:\n{allowed_animals}\n\nValidation Rules:\n1. If the image does NOT contain any animal, carcass, or animal body part (e.g., it contains only people, vehicles, buildings, landscapes, electronic devices, food, or general household items), you MUST set `is_valid_animal_image` to False and explain why in `invalid_reason` in {lang_name} language.\n2. If the image depicts an animal species NOT in the allowed list above (such as horse, snake, elephant, monkey, wild bird, fish, etc.), you MUST set `is_valid_animal_image` to False and set `invalid_reason` to 'This is not an acceptable animal, we don't process [animal name]' translated into {lang_name} language (e.g., in Lao: 'ນີ້ບໍ່ແມ່ນສັດທີ່ຮອງຮັບ, ພວກເຮົາບໍ່ໄດ້ປະມວນຜົນ [animal name]', in Thai: 'นี่ไม่ใช่สัตว์ที่รองรับ เราไม่ประมวลผล [animal name]').\n3. If and only if it is a valid animal-related image of an allowed species, set `is_valid_animal_image` to True.\n\nSTRICT DISEASE WHITELIST GUARDRAIL:\n- ONLY diseases that appear on the allowed Priority Reportable Diseases list for the identified animal species may be considered or suggested.\n- If the observed condition, symptoms, or suspected disease does NOT match any disease on the official allowed list, or if the detected disease is outside the allowed list, you MUST NOT provide any disease response (you MUST set `diseases` to an empty list `[]`). Under NO circumstances should you name or suggest any unlisted disease.",
        "prompt_template": "Analyze the provided image(s) and description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nAllowed Priority Reportable Diseases List by Animal Species:\n{allowed_diseases}\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal.\n2. Evaluate symptoms ONLY against the allowed Priority Reportable Diseases list for that animal species. If the detected/suspected disease is NOT on this list, do NOT provide any disease suggestions (return `diseases: []`).\n3. If one or more diseases from the allowed list match the symptoms, list up to three with a confidence level (0.0 to 1.0), brief reasoning/symptoms observed, and basic response recommendations for local veterinarians without requiring laboratory testing in {lang_name} language.\n4. If the confidence level is less than 0.8, you should not show the diseases suggestion, but ask the user to provide more information about the disease, perform additional lab testing, or gather more information.\n\nStrict Constraints:\n- NEVER diagnose, mention, or suggest any disease that is not explicitly on the allowed Priority Reportable Diseases list for that animal species. If not on the list, return `diseases: []`.\n- If the animal is not in the allowed list, `is_valid_animal_image` MUST be False and `invalid_reason` MUST state 'This is not an acceptable animal, we don't process [animal name]' in {lang_name} language.\n- You MUST output the entire response (including animal type, disease names, reasoning, basic_response, and invalid_reason) in {lang_name} language.\n- If a text description is provided below, incorporate it into your analysis:\n  Description: {description}"
    }

app = FastAPI(
    title="FAO-PODD Animal Disease Diagnosis API",
    description="Analyze animal images and descriptions to identify potential diseases.",
    version="2.6.3"
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

def load_symptoms():
    symptoms_path = os.path.join(os.path.dirname(__file__), "symptom.json")
    if os.path.exists(symptoms_path):
        with open(symptoms_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def normalize_animal_species(animal_str: Optional[str]) -> str:
    if not animal_str:
        return "Unknown"
    a = animal_str.strip().lower()
    mapping = {
        "chicken": "Chicken", "ไก่": "Chicken", "ໄກ່": "Chicken",
        "goose": "Goose", "ห่าน": "Goose", "ຫ່ານ": "Goose",
        "duck": "Duck", "เป็ด": "Duck", "ເປັດ": "Duck",
        "pig": "Pig", "หมู": "Pig", "สุกร": "Pig", "ໝູ": "Pig",
        "cattle": "Cattle", "cow": "Cattle", "ox": "Cattle", "bull": "Cattle", "calf": "Cattle",
        "วัว": "Cattle", "โค": "Cattle", "ງົວ": "Cattle",
        "buffalo": "Buffalo", "กระบือ": "Buffalo", "ควาย": "Buffalo", "ຄວາຍ": "Buffalo",
        "goat": "Sheep and Goat", "sheep": "Sheep and Goat", "แพะ": "Sheep and Goat",
        "แกะ": "Sheep and Goat", "ແບ້": "Sheep and Goat", "ແກະ": "Sheep and Goat",
        "dog": "Dog", "หมา": "Dog", "สุนัข": "Dog", "ໝາ": "Dog",
        "cat": "Cat", "แมว": "Cat", "ແມວ": "Cat"
    }
    for k, v in mapping.items():
        if k in a:
            return v
    return animal_str.strip()

class SymptomGuidanceMatch(BaseModel):
    symptom_no: int
    symptom_name: str
    guidance: str

class SymptomGuidanceAIResponse(BaseModel):
    matched_symptom_numbers: List[int] = Field(
        default=[],
        description="IDs of matched symptoms (1 to 46) from the official symptom list that correspond to the observed text description."
    )
    symptom_analysis: str = Field(
        description="Clinical analysis of the observed symptoms in the requested language without naming/identifying any disease."
    )
    symptom_guidance_entries: List[str] = Field(
        default=[],
        description="List of symptom guidance handbook entries for each matched symptom, fully translated/written in the requested language formatted as [Symptom Name in requested language] Specific handbook guidance and practical field care steps in requested language."
    )

def get_disease_guidelines(disease_name: str, animal_type: Optional[str] = None, lang: str = "lo") -> List[str]:
    if not disease_name:
        return []
    
    data = load_animals_and_diseases()
    target_clean = disease_name.lower().strip()
    
    # First search within specific animal if provided
    animals_to_check = []
    if animal_type:
        for a_key in data.keys():
            if a_key.lower() in animal_type.lower() or animal_type.lower() in a_key.lower():
                animals_to_check.append(a_key)
    if not animals_to_check:
        animals_to_check = list(data.keys())
        
    for a_key in animals_to_check:
        diseases = data.get(a_key, [])
        for d in diseases:
            if isinstance(d, dict):
                d_name = d.get("name", "").lower()
                d_name_lo = d.get("name_lo", "").lower()
                d_name_th = d.get("name_th", "").lower()
                
                if (target_clean in d_name or d_name in target_clean or
                    (d_name_lo and (target_clean in d_name_lo or d_name_lo in target_clean)) or
                    (d_name_th and (target_clean in d_name_th or d_name_th in target_clean))):
                    
                    resp_dict = d.get("basic_response", {})
                    if isinstance(resp_dict, dict):
                        return resp_dict.get(lang) or resp_dict.get("lo") or resp_dict.get("en") or []
    return []

class DiseaseResponse(BaseModel):
    name: str = Field(description="Name of the disease in the requested language")
    confidence: float = Field(description="Confidence level between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the symptoms and diagnosis in the requested language")
    basic_response: Optional[List[str]] = Field(default=None, description="Basic response recommendations for local veterinarians without requiring laboratory testing in the requested language")

class DiagnosisResponse(BaseModel):
    is_valid_animal_image: bool = Field(description="True if the image contains an animal or animal body parts, False otherwise")
    invalid_reason: Optional[str] = Field(None, description="Reason why the image is invalid in the requested language (if is_valid_animal_image is False)")
    animal_type: Optional[str] = Field(None, description="Type of animal identified in the requested language (null if invalid)")
    diseases: Optional[List[DiseaseResponse]] = Field(None, description="List of up to three potential diseases (null or empty if invalid)")
    symptom_guidance: Optional[List[str]] = Field(default=None, description="Guidance from symptom.json when only text description + animal type is provided without image")

def get_symptom_advice(symptom_numbers: List[int], normalized_species: str, lang: str = "lo", genai_client=None) -> List[str]:
    symptoms_catalog = load_symptoms()
    advice_items = []
    
    for s_no in symptom_numbers:
        for s in symptoms_catalog:
            if s.get("no") == s_no:
                s_name = None
                if lang == "lo":
                    s_name = s.get("symptom_name_lo") or s.get("symptom_name_en")
                elif lang == "th":
                    s_name = s.get("symptom_name_th") or s.get("symptom_name_en") or s.get("symptom_name_lo")
                else:
                    s_name = s.get("symptom_name_en") or s.get("symptom_name")
                if not s_name:
                    s_name = s.get("symptom_name") or f"Symptom #{s_no}"
                    
                sg = s.get("species_guidance", {}).get(normalized_species, {})
                guide_text = ""
                if isinstance(sg, dict):
                    if lang == "lo":
                        guide_text = sg.get("lo") or sg.get("text") or sg.get("en") or ""
                    elif lang == "en":
                        guide_text = sg.get("en") or sg.get("text") or ""
                    elif lang == "th":
                        # If explicit th is provided, use it; otherwise fallback to en (cleaner to translate or read than raw Lao)
                        guide_text = sg.get("th") or sg.get("en") or sg.get("text") or sg.get("lo") or ""
                    else:
                        guide_text = sg.get("lo") or sg.get("text") or sg.get("en") or ""
                elif isinstance(sg, str):
                    guide_text = sg
                    
                if not guide_text and normalized_species in s:
                    val = s[normalized_species]
                    if isinstance(val, str):
                        guide_text = val
                        
                guide_text = guide_text.strip()
                if guide_text:
                    advice_items.append({"s_no": s_no, "name": s_name, "text": guide_text})
                break

    if not advice_items:
        return []

    # If language is Thai, translate the handbook advice items to Thai if they are in English/Lao
    if lang == "th" and genai_client:
        try:
            translation_prompt = (
                "Translate the following veterinary symptom guidance handbook entries into Thai language accurately and professionally.\n"
                "Keep the format of each entry as [Symptom Name in Thai] Guidance text in Thai.\n"
                "Return a JSON list of strings, one string per entry.\n\n"
                "Entries to translate:\n" +
                "\n".join(f"- [{item['name']}] {item['text']}" for item in advice_items)
            )
            trans_resp = genai_client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[translation_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=List[str],
                    temperature=0.1,
                ),
            )
            translated_list = json.loads(trans_resp.text)
            if translated_list and len(translated_list) == len(advice_items):
                return translated_list
        except Exception as te:
            print(f"Error translating symptom guidance to Thai: {te}")

    return [f"[{item['name']}] {item['text']}" for item in advice_items]

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
        
    symptom_guidance = result.get("symptom_guidance")
    if symptom_guidance:
        if lang == "lo":
            guidance_header = "ຄຳແນະນຳການເບິ່ງແຍງອາການ (ອີງຕາມຄູ່ມືອາການ):"
            guidance_str = "\n".join(f"- {g}" for g in symptom_guidance)
            return f"ຜົນການວິເຄາະ AI:\n- ປະເພດສັດ: {animal_type}\n- {guidance_header}\n{guidance_str}"
        elif lang == "th":
            guidance_header = "คำแนะนำการดูแลตามอาการ (ตามคู่มืออาการ):"
            guidance_str = "\n".join(f"- {g}" for g in symptom_guidance)
            return f"ผลการวิเคราะห์ AI:\n- ประเภทสัตว์: {animal_type}\n- {guidance_header}\n{guidance_str}"
        else:
            guidance_header = "Symptom Care Guidance (from symptom handbook):"
            guidance_str = "\n".join(f"- {g}" for g in symptom_guidance)
            return f"AI Analysis Result:\n- Animal Type: {animal_type}\n- {guidance_header}\n{guidance_str}"

    diseases_list = []
    for d in (result.get("diseases") or []):
        name = d.get("name")
        conf = d.get("confidence", 0.0) * 100
        reason = d.get("reasoning") or ""
        basic_resp = d.get("basic_response") or []
        
        disease_entry = f"- {name} ({conf:.0f}%): {reason}"
        if basic_resp:
            if lang == "lo":
                guidelines_header = "  ຄຳແນະນຳເບື້ອງຕົ້ນ:"
            elif lang == "th":
                guidelines_header = "  คำแนะนำเบื้องต้น:"
            else:
                guidelines_header = "  Basic Guidelines:"
            resp_lines = "\n".join(f"    * {r}" for r in basic_resp)
            disease_entry += f"\n{guidelines_header}\n{resp_lines}"
            
        diseases_list.append(disease_entry)
        
    if not diseases_list:
        if lang == "lo":
            diseases_str = "- ບໍ່ພົບພະຍາດໃນລາຍການແຈ້ງເຕືອນບູລິມະສິດ"
        elif lang == "th":
            diseases_str = "- ไม่พบโรคในรายการที่กำหนด"
        else:
            diseases_str = "- None detected from priority reportable list"
    else:
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

async def fetch_lahis_incident_detail(report_id: str, token: str, api_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        incident_url = f"{api_url.rstrip('/')}/api/integrations/v1/incidents/{report_id}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = await client.get(incident_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch incident {report_id} detail from LAHIS: {str(e)}"
            )

async def submit_lahis_cluster(cluster_payload: dict, token: str, api_url: str) -> dict:
    async with httpx.AsyncClient() as client:
        cluster_url = f"{api_url.rstrip('/')}/api/integrations/v1/clusters"
        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": cluster_payload.get("externalClusterId", f"cluster-{uuid.uuid4()}"),
            "Content-Type": "application/json"
        }
        try:
            response = await client.post(cluster_url, json=cluster_payload, headers=headers, timeout=15.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"LAHIS cluster API returned error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to submit cluster result to LAHIS: {str(e)}"
            )

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_animal_image(
    request: Request,
    images: Optional[List[UploadFile]] = File(None, description="Images of the sick/diseased animal"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID to fetch the image from"),
    description: Optional[str] = Form(None, description="Optional text description of signs/symptoms"),
    animal_type: Optional[str] = Form(None, description="Animal type / species when submitting without image"),
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
        "animal_type": animal_type,
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
            if not animal_type and "animal_type" in body:
                animal_type = body.get("animal_type")
        else:
            request_params = {
                "lang": lang,
                "description": description,
                "animal_type": animal_type,
                "report_id": report_id,
                "image_filenames": [img.filename for img in images] if images else [],
                "source": request.headers.get("x-source", "api")
            }
        steps["step_1"]["data"] = request_params
        print(f"[STEP 1] Request from LAHIS. Content-Type: {content_type}, is_json: {is_json}, params/body: {json.dumps(request_params)}")
        if lang not in ["en", "th", "lo"]:
            raise HTTPException(status_code=400, detail="Unsupported language. Supported languages are 'en', 'th', 'lo'.")
            
        if not is_json and not images and not report_id and not (description and animal_type):
            raise HTTPException(status_code=400, detail="Either image files or report_id must be provided. Alternatively, provide description and animal_type for text symptom consultation.")
        
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
        elif report_id:
            # Fetch image bytes from LAHIS API
            if not all([api_url, client_id, client_secret]):
                raise HTTPException(
                    status_code=500,
                    detail="LAHIS integration environment variables (TENANT_API_URL, LAHIS_CLIENT_ID, LAHIS_CLIENT_SECRET) are not fully configured."
                )
                
            token = await get_lahis_token(client_id, client_secret, api_url)
            multiple_images_bytes = await fetch_lahis_report_images(report_id, token, api_url)
            
        has_images = len(multiple_images_bytes) > 0

        if not has_images:
            # Text-only consultation using symptom.json without identifying any disease
            if not description or not animal_type:
                raise HTTPException(status_code=400, detail="When submitting without image, both description and animal_type are required.")
                
            norm_animal = normalize_animal_species(animal_type)
            allowed_animals = ['Buffalo', 'Cat', 'Cattle', 'Chicken', 'Dog', 'Duck', 'Goose', 'Pig', 'Sheep and Goat']
            
            lang_names = {"en": "English", "th": "Thai", "lo": "Lao"}
            lang_name = lang_names.get(lang, "English")
            
            if norm_animal not in allowed_animals:
                invalid_msg = {
                    "lo": f"ນີ້ບໍ່ແມ່ນສັດທີ່ຮອງຮັບ, ພວກເຮົາບໍ່ໄດ້ປະມວນຜົນ {animal_type}",
                    "th": f"นี่ไม่ใช่สัตว์ที่รองรับ เราไม่ประมวลผล {animal_type}",
                    "en": f"This is not an acceptable animal, we don't process {animal_type}"
                }.get(lang, f"This is not an acceptable animal, we don't process {animal_type}")
                result = {
                    "is_valid_animal_image": False,
                    "invalid_reason": invalid_msg,
                    "animal_type": animal_type,
                    "diseases": [],
                    "symptom_guidance": []
                }
                steps["step_2"] = {
                    "title": "2. Animal Validation",
                    "data": {"animal_type": animal_type, "normalized": norm_animal, "valid": False}
                }
                steps["step_3"] = {
                    "title": "3. Validation Result",
                    "data": result
                }
            else:
                genai_client = get_genai_client()
                symptoms_catalog = load_symptoms()
                symptom_lines = []
                for s in symptoms_catalog:
                    s_no = s.get("no")
                    s_en = s.get("symptom_name_en")
                    s_lo = s.get("symptom_name_lo")
                    symptom_lines.append(f"{s_no}. {s_en} ({s_lo})")
                symptoms_list_str = "\n".join(symptom_lines)
                
                text_prompt = f"""You are an expert veterinary clinical symptom analyzer.
The user provided a clinical description of a sick animal without any image.
Animal Species: {animal_type} (Normalized: {norm_animal})
Description of observed signs and symptoms:
{description}

Here is the official reference list of 46 clinical symptoms:
{symptoms_list_str}

Tasks:
1. Identify which symptoms from the official reference list match the observed signs described in the text description.
   Return their symptom numbers in `matched_symptom_numbers` (an empty list if none match).
2. In `symptom_analysis`, provide a clear, supportive clinical assessment of the reported symptoms in {lang_name} language.
3. For each matched symptom, look up its standard veterinary field handbook guidance and provide it in `symptom_guidance_entries` translated and written entirely in {lang_name} language.
   Format each entry as: [Symptom Name in {lang_name}] Specific handbook guidance and practical field care steps in {lang_name} language.

STRICT CONSTRAINTS:
- DO NOT name, diagnose, suspect, or identify any specific disease under any circumstances (e.g., do NOT mention FMD, Anthrax, Swine Fever, Rabies, etc.).
- Focus strictly on the symptoms, clinical signs, and condition management.
- All entries in `symptom_analysis` and `symptom_guidance_entries` MUST be written in {lang_name} language.
- Return your response strictly matching the schema."""

                steps["step_2"] = {
                    "title": "2. Request to Google AI (Symptom Consultation)",
                    "data": {
                        "model": "gemini-3.5-flash",
                        "prompt": text_prompt
                    }
                }
                print(f"[STEP 2] Request to Google AI (Text Symptom). Model: gemini-3.5-flash, Prompt length: {len(text_prompt)}")
                
                response = genai_client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=[text_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SymptomGuidanceAIResponse,
                        temperature=0.2,
                    ),
                )
                print(f"[STEP 3] Response from Google AI: {response.text}")
                prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                ai_parsed = json.loads(response.text)
                
                matched_nos = ai_parsed.get("matched_symptom_numbers", [])
                symptom_analysis = ai_parsed.get("symptom_analysis", "")
                ai_guidance_entries = ai_parsed.get("symptom_guidance_entries", [])
                
                # If Gemini provided guidance entries in the requested language, use them;
                # otherwise fall back to localized get_symptom_advice.
                if ai_guidance_entries and len(ai_guidance_entries) > 0:
                    advice_list = ai_guidance_entries
                else:
                    advice_list = get_symptom_advice(matched_nos, norm_animal, lang, genai_client=genai_client)
                
                full_guidance = []
                if symptom_analysis:
                    full_guidance.append(symptom_analysis)
                full_guidance.extend(advice_list)
                
                result = {
                    "is_valid_animal_image": True,
                    "invalid_reason": None,
                    "animal_type": animal_type,
                    "diseases": [],
                    "symptom_guidance": full_guidance
                }
                
                steps["step_3"] = {
                    "title": "3. Response from Google AI (Symptom Consultation)",
                    "data": {
                        "result": result,
                        "matched_symptom_numbers": matched_nos,
                        "usage_metadata": {
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens
                        }
                    }
                }
        else:
            pil_images = []
            for i, img_bytes in enumerate(multiple_images_bytes):
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    pil_images.append(pil_img)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to parse image index {i}: {str(e)}")
            
            # Save images for audit (locally + Google Cloud Storage)
            saved_paths = []
            for i, img_bytes in enumerate(multiple_images_bytes):
                try:
                    ext = ".jpg"
                    pil_img = pil_images[i]
                    if pil_img.format:
                        ext = f".{pil_img.format.lower()}"
                    saved_image_filename = f"{uuid.uuid4()}{ext}"
                    c_type = f"image/{ext.lstrip('.')}" if ext != ".jpg" else "image/jpeg"
                    save_audit_image(img_bytes, saved_image_filename, content_type=c_type)
                    saved_paths.append(f"logs/images/{saved_image_filename}")
                except Exception as e:
                    print(f"Error saving image {i} to logs: {e}")
            if saved_paths:
                saved_image_path = ",".join(saved_paths)
            
            # Get initialized Gemini Client
            genai_client = get_genai_client()
            
            # Define prompt based on language
            lang_names = {"en": "English", "th": "Thai", "lo": "Lao"}
            lang_name = lang_names.get(lang, "English")
            
            # Load guardrail policy and prompt_template from config.json
            config = load_config()
            animals_and_diseases = load_animals_and_diseases()
            allowed_animals_text = "\n".join(f"- {animal}" for animal in animals_and_diseases.keys())
            disease_lines = []
            for animal, diseases in animals_and_diseases.items():
                if diseases and isinstance(diseases[0], dict):
                    names = [d.get("name", "") for d in diseases if isinstance(d, dict)]
                else:
                    names = diseases
                disease_lines.append(f"- {animal}: {', '.join(names)}")
            allowed_diseases_text = "\n".join(disease_lines)

            guardrail_policy = config.get("guardrail_policy", "")
            if "{allowed_animals}" in guardrail_policy:
                guardrail_policy = guardrail_policy.replace("{allowed_animals}", allowed_animals_text)

            prompt_template = config.get(
                "prompt_template",
                "Analyze the provided image(s) and description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nAllowed Priority Reportable Diseases List by Animal Species:\n{allowed_diseases}\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal.\n2. Evaluate symptoms ONLY against the allowed Priority Reportable Diseases list for that animal species. If the detected/suspected disease is NOT on this list, do NOT provide any disease suggestions (return `diseases: []`).\n3. If one or more diseases from the allowed list match the symptoms, list up to three with a confidence level (0.0 to 1.0) and brief reasoning/symptoms observed.\n4. If the confidence level is less than 0.8, you should not show the diseases suggestion, but ask the user to provide more information about the disease, perform additional lab testing, or gather more information.\n\nStrict Constraints:\n- NEVER diagnose, mention, or suggest any disease that is not explicitly on the allowed Priority Reportable Diseases list for that animal species. If not on the list, return `diseases: []`.\n- If the animal is not in the allowed list, `is_valid_animal_image` MUST be False and `invalid_reason` MUST state 'This is not an acceptable animal, we don't process [animal name]' in {lang_name} language.\n- You MUST output the entire response (including animal type, disease names, reasoning, and invalid_reason) in {lang_name} language.\n- If a text description is provided below, incorporate it into your analysis:\n  Description: {description}"
            )
            if "{allowed_diseases}" in prompt_template:
                prompt_template = prompt_template.replace("{allowed_diseases}", allowed_diseases_text)
            if "{allowed_animals}" in prompt_template:
                prompt_template = prompt_template.replace("{allowed_animals}", allowed_animals_text)

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
            
            # Enrich disease items with verified basic_response guidelines if not populated
            if result.get("diseases"):
                for d in result["diseases"]:
                    if not d.get("basic_response"):
                        d["basic_response"] = get_disease_guidelines(
                            disease_name=d.get("name", ""),
                            animal_type=result.get("animal_type"),
                            lang=lang
                        )
                        
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

@app.post("/detect-cluster")
async def detect_cluster(
    request: Request,
    images: Optional[List[UploadFile]] = File(None, description="Optional images"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID"),
    description: Optional[str] = Form(None, description="Optional text description"),
    lang: str = Form("en", description="Language for response: 'en', 'th', 'lo'")
):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    request_params = {
        "lang": lang,
        "description": description,
        "report_id": report_id,
        "image_filenames": [img.filename for img in images] if images else [],
        "source": request.headers.get("x-source", "api")
    }
    steps = {
        "step_1": {
            "title": "1. Request Received",
            "data": request_params
        }
    }
    
    try:
        content_type = request.headers.get("content-type", "")
        is_json = "application/json" in content_type
        event_id = "unknown-event"
        raw_body_bytes = b""
        
        if is_json:
            raw_body_bytes = await request.body()
            try:
                body = json.loads(raw_body_bytes)
                request_params = body
            except Exception:
                request_params = {"raw_body": raw_body_bytes.decode("utf-8", errors="ignore")}
                raise HTTPException(status_code=400, detail="Invalid JSON body")
                
            timestamp = request.headers.get("x-ohtk-timestamp", "")
            signature = request.headers.get("x-ohtk-signature", "")
            
            signing_secret = os.getenv("LAHIS_SIGNING_SECRET")
            if signing_secret:
                if not signature or not timestamp:
                    raise HTTPException(status_code=401, detail="Missing signature headers")
                
                path = request.url.path
                alt_path = path + "/" if not path.endswith("/") else path[:-1]
                
                sig_ok = verify_webhook_signature(path, timestamp, raw_body_bytes, signature)
                if not sig_ok:
                    sig_ok = verify_webhook_signature(alt_path, timestamp, raw_body_bytes, signature)
                if not sig_ok:
                    raise HTTPException(status_code=401, detail="Invalid webhook signature")
            
            report_data = body.get("report") or {}
            report_id = report_data.get("id")
            if not report_id:
                raise HTTPException(status_code=400, detail="Missing report.id in payload")
            event_id = body.get("eventId") or "unknown-event"
            
        if not report_id:
            raise HTTPException(status_code=400, detail="report_id is required for cluster detection.")
            
        api_url = os.getenv("TENANT_API_URL")
        client_id = os.getenv("LAHIS_CLIENT_ID")
        client_secret = os.getenv("LAHIS_CLIENT_SECRET")
        
        if not all([api_url, client_id, client_secret]):
            raise HTTPException(
                status_code=500,
                detail="LAHIS integration environment variables (TENANT_API_URL, LAHIS_CLIENT_ID, LAHIS_CLIENT_SECRET) are not fully configured."
            )
            
        # Step 2: Get OAuth token & fetch incident detail from LAHIS
        token = await get_lahis_token(client_id, client_secret, api_url)
        incident_data = await fetch_lahis_incident_detail(report_id, token, api_url)
        
        steps["step_2"] = {
            "title": "2. Fetched Incident Detail from LAHIS",
            "data": incident_data
        }
        
        incident = incident_data.get("incident") or incident_data
        
        # Extract village information and date
        village_info = incident.get("village") or {}
        village_id = village_info.get("id") or incident.get("villageId") or incident.get("village_id")
        village_name = village_info.get("name") or incident.get("villageName") or incident.get("village_name")
        
        if not village_id:
            relevant_authorities = incident.get("relevantAuthorityIds") or []
            if relevant_authorities:
                village_id = f"auth-{relevant_authorities[0]}"
            elif incident.get("location"):
                loc = incident.get("location")
                village_id = f"loc-{loc.get('lat')},{loc.get('lon')}"
            else:
                village_id = "default_village"
                
        incident_date = incident.get("incidentDate") or datetime.utcnow().strftime("%Y-%m-%d")
        
        # Record this incoming request to DB
        record_cluster_report(
            report_id=report_id,
            village_id=str(village_id),
            village_name=village_name,
            date_str=incident_date,
            raw_payload=json.dumps(request_params)
        )
        
        # Check all reports for this village across history
        all_village_reports = get_village_all_reports(str(village_id))
        report_count = len(all_village_reports)
        
        # Calculate timeline span for this village
        report_dates = []
        for r in all_village_reports:
            d_str = r.get("date_str")
            if d_str:
                try:
                    dt = datetime.strptime(d_str, "%Y-%m-%d")
                    report_dates.append(dt)
                except Exception:
                    pass
                    
        if report_dates:
            earliest_date = min(report_dates)
            latest_date = max(report_dates)
            span_days = (latest_date - earliest_date).days
        else:
            earliest_date = datetime.utcnow()
            latest_date = datetime.utcnow()
            span_days = 0
            
        cluster_detected = span_days > 14
        cluster_response = None
        
        steps["step_3"] = {
            "title": "3. Timeline Cluster Check",
            "data": {
                "village_id": str(village_id),
                "village_name": village_name,
                "earliest_date": earliest_date.strftime("%Y-%m-%d"),
                "latest_date": latest_date.strftime("%Y-%m-%d"),
                "span_days": span_days,
                "report_count": report_count,
                "cluster_detected": cluster_detected
            }
        }
        
        if cluster_detected:
            # Collect unique incident IDs in this village
            all_incident_ids = list(set([r["report_id"] for r in all_village_reports if r.get("report_id")]))
            if report_id not in all_incident_ids:
                all_incident_ids.append(report_id)
                
            village_id_int = int(village_id) if str(village_id).isdigit() else None
            authority_ids = incident.get("relevantAuthorityIds") or []
            
            cluster_payload = {
                "externalClusterId": f"cluster-{village_id}-{earliest_date.strftime('%Y%m%d')}-{latest_date.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
                "algorithmVersion": "podd-cluster-detector-v2",
                "window": {
                    "from": earliest_date.strftime("%Y-%m-%d"),
                    "to": latest_date.strftime("%Y-%m-%d")
                },
                "incidentIds": all_incident_ids,
                "authorityIds": authority_ids,
                "villageIds": [village_id_int] if village_id_int is not None else [],
                "geometry": incident.get("location"),
                "radiusMeters": 500.0,
                "score": 1.0,
                "riskLevel": "HIGH",
                "explanation": f"Cluster detected: Suspected disease reports in village {village_name or village_id} continued for {span_days} days (>14 days).",
                "metadata": {
                    "source": "fao-podd-cluster-detector",
                    "span_days": span_days,
                    "report_count": report_count
                }
            }
            
            steps["step_4"] = {
                "title": "4. Sending Cluster to LAHIS Callback",
                "data": cluster_payload
            }
            
            try:
                cluster_response = await submit_lahis_cluster(cluster_payload, token, api_url)
                steps["step_5"] = {
                    "title": "5. LAHIS Callback Response",
                    "data": cluster_response
                }
            except HTTPException as e:
                steps["step_5"] = {
                    "title": "5. LAHIS Callback Response (Error)",
                    "data": {"status_code": e.status_code, "detail": e.detail}
                }
                
        result = {
            "status": "success",
            "cluster_detected": cluster_detected,
            "report_count": report_count,
            "village_id": str(village_id),
            "village_name": village_name,
            "earliest_date": earliest_date.strftime("%Y-%m-%d"),
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "span_days": span_days,
            "cluster_response": cluster_response
        }
        
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/detect-cluster",
            status_code=200,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps(result),
            steps_json=steps
        )
        return result

    except HTTPException as e:
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/detect-cluster",
            status_code=e.status_code,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": e.detail}),
            steps_json=steps
        )
        raise e
    except Exception as e:
        latency = time.time() - start_time
        log_request_response(
            method="POST",
            path="/detect-cluster",
            status_code=500,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": str(e)}),
            steps_json=steps
        )
        raise HTTPException(status_code=500, detail=f"Error detecting cluster: {str(e)}")

def load_symptoms():
    symptoms_path = os.path.join(os.path.dirname(__file__), "symptom.json")
    if os.path.exists(symptoms_path):
        with open(symptoms_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.get("/api/diseases")
async def get_diseases():
    return load_animals_and_diseases()

@app.get("/api/symptoms")
async def get_symptoms():
    return load_symptoms()


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(username: str = Depends(authenticate_dashboard)):
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>FAO-PODD Dashboard</h1><p>dashboard.html not found</p>"

@app.get("/privacy", response_class=HTMLResponse)
async def serve_privacy_policy():
    privacy_path = os.path.join(os.path.dirname(__file__), "privacy.html")
    if os.path.exists(privacy_path):
        with open(privacy_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>FAO-PODD Privacy Policy</h1><p>privacy.html not found</p>"

@app.get("/home", response_class=HTMLResponse)
@app.get("/about", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_landing_page():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>FAO-PODD Webpage</h1><p>index.html not found</p>"



def fetch_all_logs_data(limit: Optional[int] = 50):
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ApiLog")
            query.order = ["-timestamp"]
            fetch_limit = limit if (limit and limit > 0) else None
            results = list(query.fetch(limit=fetch_limit))
            
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
        if limit and limit > 0:
            cursor.execute("SELECT * FROM api_logs ORDER BY id DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM api_logs ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            raw_params = row["request_params"]
            try:
                parsed_params = json.loads(raw_params) if raw_params else {}
            except Exception:
                parsed_params = {"raw": raw_params}

            raw_steps = row["steps_json"] if "steps_json" in row.keys() else None
            try:
                parsed_steps = json.loads(raw_steps) if raw_steps else None
            except Exception:
                parsed_steps = None

            logs.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "method": row["method"],
                "path": row["path"],
                "status_code": row["status_code"],
                "latency": row["latency"],
                "client_ip": row["client_ip"],
                "request_params": parsed_params,
                "response_body": row["response_body"],
                "steps_json": parsed_steps,
                "image_path": row["image_path"] if ("image_path" in row.keys() and row["image_path"]) else None,
                "prompt_tokens": row["prompt_tokens"] if "prompt_tokens" in row.keys() else 0,
                "output_tokens": row["output_tokens"] if "output_tokens" in row.keys() else 0
            })
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")


@app.get("/api/logs")
async def get_logs(limit: Optional[int] = 50, username: str = Depends(authenticate_dashboard)):
    return fetch_all_logs_data(limit=limit)


@app.get("/api/logs/export")
async def export_logs(format: str = "json", username: str = Depends(authenticate_dashboard)):
    logs = fetch_all_logs_data(limit=None)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format.lower() == "csv":
        output = io.StringIO()
        fieldnames = [
            "id", "timestamp", "method", "path", "status_code", "latency",
            "client_ip", "prompt_tokens", "output_tokens", "target",
            "image_path", "request_params", "response_body"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            req_params = log.get("request_params") or {}
            target = "None"
            if isinstance(req_params, dict):
                if req_params.get("report_id"):
                    target = f"Report ID: {req_params['report_id']}"
                elif req_params.get("report", {}).get("id"):
                    target = f"Report ID: {req_params['report']['id']}"
                elif req_params.get("image_filenames"):
                    target = f"Uploads: {', '.join(req_params['image_filenames'])}"
                elif req_params.get("image_filename"):
                    target = f"Upload: {req_params['image_filename']}"

            writer.writerow({
                "id": log.get("id"),
                "timestamp": log.get("timestamp"),
                "method": log.get("method"),
                "path": log.get("path"),
                "status_code": log.get("status_code"),
                "latency": log.get("latency"),
                "client_ip": log.get("client_ip"),
                "prompt_tokens": log.get("prompt_tokens", 0),
                "output_tokens": log.get("output_tokens", 0),
                "target": target,
                "image_path": log.get("image_path") or "",
                "request_params": json.dumps(req_params, ensure_ascii=False) if isinstance(req_params, dict) else str(req_params),
                "response_body": log.get("response_body") or ""
            })

        filename = f"api_logs_{timestamp_str}.csv"
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    else:
        # Default to JSON export
        content = json.dumps(logs, indent=2, ensure_ascii=False)
        filename = f"api_logs_{timestamp_str}.json"
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )


@app.get("/api/clusters")
async def get_clusters(limit: int = 100, username: str = Depends(authenticate_dashboard)):
    reports = []
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ClusterReport")
            query.order = ["-timestamp"]
            results = list(query.fetch(limit=limit))
            for entity in results:
                reports.append({
                    "id": entity.key.id,
                    "timestamp": entity.get("timestamp"),
                    "date_str": entity.get("date_str"),
                    "report_id": entity.get("report_id"),
                    "village_id": entity.get("village_id"),
                    "village_name": entity.get("village_name"),
                    "raw_payload": entity.get("raw_payload")
                })
        except Exception as e:
            print(f"Datastore fetch cluster reports error: {e}")

    if not reports:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cluster_reports ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            for row in rows:
                reports.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "date_str": row["date_str"],
                    "report_id": row["report_id"],
                    "village_id": row["village_id"],
                    "village_name": row["village_name"],
                    "raw_payload": row["raw_payload"]
                })
        except Exception as e:
            print(f"SQLite fetch cluster reports error: {e}")

    # Group by village_id
    groups = {}
    for r in reports:
        vid = r["village_id"]
        if vid not in groups:
            groups[vid] = {
                "village_id": vid,
                "village_name": r["village_name"] or "Unknown Village",
                "reports": [],
                "dates": [],
                "latest_timestamp": r["timestamp"]
            }
        if r["report_id"] and r["report_id"] not in groups[vid]["reports"]:
            groups[vid]["reports"].append(r["report_id"])
        
        d_str = r.get("date_str")
        if d_str:
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                groups[vid]["dates"].append(dt)
            except Exception:
                pass

    grouped_list = []
    total_clusters = 0

    for vid, g in groups.items():
        if g["dates"]:
            min_d = min(g["dates"])
            max_d = max(g["dates"])
            span_days = (max_d - min_d).days
            start_date_str = min_d.strftime("%Y-%m-%d")
            latest_date_str = max_d.strftime("%Y-%m-%d")
        else:
            span_days = 0
            start_date_str = "N/A"
            latest_date_str = "N/A"

        is_cluster = span_days > 14
        if is_cluster:
            total_clusters += 1

        grouped_list.append({
            "village_id": vid,
            "village_name": g["village_name"],
            "start_date": start_date_str,
            "latest_date": latest_date_str,
            "span_days": span_days,
            "report_count": len(g["reports"]),
            "reports": g["reports"],
            "cluster_detected": is_cluster,
            "latest_timestamp": g["latest_timestamp"]
        })

    grouped_list.sort(key=lambda x: (x["latest_timestamp"] or ""), reverse=True)

    return {
        "stats": {
            "total_reports": len(reports),
            "unique_villages": len(groups),
            "total_clusters": total_clusters
        },
        "clusters": grouped_list,
        "raw_reports": reports
    }

@app.get("/api/logs/image/{filename}")
async def get_log_image(filename: str, username: str = Depends(authenticate_dashboard)):
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    img_bytes = get_audit_image_bytes(filename)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="Image not found")
        
    media_type = "image/jpeg"
    if filename.lower().endswith(".png"):
        media_type = "image/png"
    elif filename.lower().endswith(".webp"):
        media_type = "image/webp"
        
    return Response(content=img_bytes, media_type=media_type)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port)

