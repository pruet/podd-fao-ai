import os
import io
import hmac
import hashlib
import re
from typing import Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
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

from tenant import (
    TenantContext,
    TenantSigningKey,
    TenantRegistry,
    create_default_registry,
    normalize_api_base_url,
    TenantTokenManager,
    LahisTenantClient,
    validate_webhook_request,
    verify_hmac_signature,
    validate_timestamp_freshness
)
from storage import (
    JobRecord,
    init_sqlite_db,
    record_cluster_report,
    get_village_reports_for_date,
    get_village_all_reports,
    log_request_response,
    get_or_create_job,
    update_job,
    DB_PATH,
    LOGS_DIR,
    USE_DATASTORE,
    IS_GAE
)

tenant_registry = create_default_registry()
tenant_token_manager = TenantTokenManager()
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
    init_sqlite_db()

# Initialize DB on load
init_db()

def get_ai_diagnosed_diseases_for_report(report_id: str, tenant_key: Optional[str] = None) -> List[str]:
    if not report_id:
        return []
    diseases_found = []

    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ApiLog")
            query.add_filter("path", "=", "/analyze")
            if tenant_key:
                query.add_filter("tenant_key", "=", str(tenant_key))
            query.order = ["-timestamp"]
            results = list(query.fetch(limit=100))
            for entity in results:
                raw_params = entity.get("request_params") or ""
                if report_id in str(raw_params):
                    resp_body = entity.get("response_body") or ""
                    try:
                        parsed = json.loads(resp_body)
                        for d in (parsed.get("diseases") or []):
                            if isinstance(d, dict) and d.get("name"):
                                diseases_found.append(d["name"])
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error fetching AI diagnosed diseases from Datastore: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if tenant_key:
            cursor.execute(
                "SELECT response_body FROM api_logs WHERE path = '/analyze' AND tenant_key = ? AND request_params LIKE ? ORDER BY id DESC LIMIT 10",
                (str(tenant_key), f"%{report_id}%")
            )
        else:
            cursor.execute(
                "SELECT response_body FROM api_logs WHERE path = '/analyze' AND request_params LIKE ? ORDER BY id DESC LIMIT 10",
                (f"%{report_id}%",)
            )
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            resp_body = row["response_body"]
            if resp_body:
                try:
                    parsed = json.loads(resp_body)
                    for d in (parsed.get("diseases") or []):
                        if isinstance(d, dict) and d.get("name"):
                            diseases_found.append(d["name"])
                except Exception:
                    pass
    except Exception as e:
        print(f"Error fetching AI diagnosed diseases from SQLite: {e}")

    return list(dict.fromkeys(diseases_found))

def is_cluster_trigger_disease(disease_name: Optional[str], config: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    if not disease_name:
        return False, None

    if config is None:
        config = load_config()

    trigger_list = config.get("cluster_trigger_diseases", [
        "Foot-and-mouth disease (FMD)",
        "African swine fever (ASF)",
        "Highly pathogenic avian influenza (HPAI)",
        "Newcastle disease",
        "Rabies",
        "Anthrax",
        "Lumpy skin disease (LSD)"
    ])

    name_clean = disease_name.strip().lower()

    aliases = {
        "Foot-and-mouth disease (FMD)": [
            "foot-and-mouth disease", "foot and mouth disease", "foot and mouth",
            "foot-and-mouth", "fmd", "ปากและเท้าเปื่อย", "โรคปากและเท้าเปื่อย",
            "ປາກ ແລະ ເລັບເປື່ອຍ", "ພະຍາດປາກ ແລະ ເລັບເປື່ອຍ", "ປາກເປື່ອຍເລັບເປື່ອຍ"
        ],
        "African swine fever (ASF)": [
            "african swine fever", "asf", "อหิวาต์แอฟริกา", "อหิวาต์แอฟริกาในสุกร",
            "โรคอหิวาต์แอฟริกาในสุกร", "ອະຫິວາໝູອາຟຣິກາ", "ພະຍາດອະຫິວາໝູອາຟຣິກາ"
        ],
        "Highly pathogenic avian influenza (HPAI)": [
            "highly pathogenic avian influenza", "avian influenza", "hpai",
            "bird flu", "avian flu", "ไข้หวัดนก", "โรคไข้หวัดนก",
            "ໄຂ້ຫວັດສັດປີກ", "ພະຍາດໄຂ້ຫວັດສັດປີກ"
        ],
        "Newcastle disease": [
            "newcastle disease", "newcastle", "nd", "ห่าไก่", "โรคห่าไก่",
            "นิวคาสเซิล", "โรคนิวคาสเซิล", "ຫ່າໄກ່", "ພະຍາດຫ່າໄກ່",
            "ນິວຄາສເຊິລ", "ພະຍາດນິວຄາສເຊິລ"
        ],
        "Rabies": [
            "rabies", "พิษสุนัขบ้า", "โรคพิษสุนัขบ้า", "โรคกลัวน้ำ",
            "ວໍ້", "ພະຍາດວໍ້"
        ],
        "Anthrax": [
            "anthrax", "แอนแทรกซ์", "โรคแอนแทรกซ์", "กาลี",
            "ແອນແທຣກ", "ພະຍາດແອນແທຣກ"
        ],
        "Lumpy skin disease (LSD)": [
            "lumpy skin disease", "lumpy skin", "lsd", "ลัมปี สกิน", "ลัมปีสกิน",
            "โรคลัมปี สกิน", "ລຳປີສະກິນ", "ຜິວໜັງອັກເສບຕິດຕໍ່",
            "ພະຍາດຜິວໜັງອັກເສບຕິດຕໍ່ (lumpy skin)"
        ]
    }

    # 1. Match against configured trigger list using known aliases and direct names
    for trigger in trigger_list:
        trigger_lower = trigger.strip().lower()
        if trigger_lower in name_clean or name_clean in trigger_lower:
            return True, trigger

        for canonical, alias_keywords in aliases.items():
            if canonical.lower() == trigger_lower or trigger_lower in canonical.lower():
                for kw in alias_keywords:
                    if kw in name_clean:
                        return True, trigger

    # 2. General fallback token match
    for trigger in trigger_list:
        t_tokens = set(re.findall(r"\w+", trigger.lower()))
        n_tokens = set(re.findall(r"\w+", name_clean))
        stopwords = {"disease", "of", "and", "in", "the", "โรค", "ພະຍາດ"}
        t_sig = t_tokens - stopwords
        n_sig = n_tokens - stopwords
        if t_sig and n_sig and t_sig.issubset(n_sig):
            return True, trigger

    return False, None


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
        "guardrail_policy": "Determine if the request involves an animal belonging to the ALLOWED animal species list:\n{allowed_animals}\n\nValidation Rules:\n1. For image-based diagnosis: Determine if the image depicts a living animal, animal carcass, or animal body parts (such as skin, mouth, organs, legs, etc.) of an allowed animal species. If the image does NOT contain any animal, carcass, or animal body part (e.g., contains only people, vehicles, buildings, landscapes, electronic devices, food, or general household items), you MUST set `is_valid_animal_image` to False and explain why in `invalid_reason` in {lang_name} language.\n2. For animal species validation (both image analysis and text-only symptom consultations): If the animal species is NOT in the allowed list above (such as horse, snake, elephant, monkey, wild bird, fish, etc.), you MUST set `is_valid_animal_image` to False and set `invalid_reason` to 'This is not an acceptable animal, we don't process [animal name]' translated into {lang_name} language (e.g., in Lao: 'ນີ້ບໍ່ແມ່ນສັດທີ່ຮອງຮັບ, ພວກເຮົາບໍ່ໄດ້ປະມວນຜົນ [animal name]', in Thai: 'นี่ไม่ใช่สัตว์ที่รองรับ เราไม่ประมวลผล [animal name]').\n3. If and only if it is a valid animal of an allowed species, set `is_valid_animal_image` to True.\n\nSTRICT DISEASE WHITELIST GUARDRAIL (Image-Based Diagnosis):\n- ONLY diseases that appear on the allowed Priority Reportable Diseases list for the identified animal species may be considered or suggested.\n- If the observed condition, symptoms, or suspected disease does NOT match any disease on the official allowed list, or if the detected disease is outside the allowed list, you MUST NOT provide any disease response (you MUST set `diseases` to an empty list `[]`). Under NO circumstances should you name or suggest any unlisted disease.\n\nSTRICT SYMPTOM-ONLY CONSULTATION GUARDRAIL (Text-Only Requests Without Images):\n- When analyzing clinical descriptions without images, DO NOT diagnose, name, suspect, or identify any specific disease under any circumstances.\n- Focus strictly on matching observed signs against the official reference symptoms handbook and providing condition management guidance.",
        "prompt_template": "Analyze the provided image(s) and clinical text description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nAllowed Priority Reportable Diseases List by Animal Species:\n{allowed_diseases}\n\nClinical Context & Reporter Information:\n- Text Description / Observed Clinical Signs: {description}\n- Stated Animal Species (if provided): {animal_species_context}\n\nMultimodal Diagnostic Guidelines:\n- MULTIMODAL SYNTHESIS: Perform a comprehensive multimodal veterinary evaluation by combining BOTH the visual evidence in the image(s) (e.g., visible lesions, blisters, vesicles, ulcers, erosions, discoloration, swellings, discharges, body posture, skin/mucosal changes) AND the clinical observations or history provided in the text description (e.g., high fever, sudden death, lameness, anorexia, salivation, diarrhea, coughing, abortion, progression rate, herd mortality).\n- The text description provides vital clinical context: use it alongside the image(s) to corroborate visual findings, differentiate diseases that share similar visual lesions, and assess overall disease severity.\n- If no text description is provided or if it states 'None provided', base your clinical evaluation solely on the visual findings in the image(s).\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal (confirming against the stated animal species if provided).\n2. Multimodal Disease Evaluation: Evaluate the combined visual symptoms from the image(s) and clinical signs from the description ONLY against the allowed Priority Reportable Diseases list for that animal species. If the detected/suspected disease is NOT on this list, do NOT provide any disease suggestions (return `diseases: []`).\n3. If one or more diseases from the allowed list match the combined visual and described symptoms, list up to three with a confidence level (0.0 to 1.0), reasoning explaining how both the visual lesions and the described clinical signs support the diagnosis, and basic response recommendations for local veterinarians without requiring laboratory testing in {lang_name} language.\n4. If the confidence level is less than 0.8, you should not show the diseases suggestion, but ask the user to provide more information about the disease, perform additional lab testing, or gather more information.\n\nStrict Constraints:\n- NEVER diagnose, mention, or suggest any disease that is not explicitly on the allowed Priority Reportable Diseases list for that animal species. If not on the list, return `diseases: []`.\n- If the animal is not in the allowed list, `is_valid_animal_image` MUST be False and `invalid_reason` MUST state 'This is not an acceptable animal, we don't process [animal name]' in {lang_name} language.\n- You MUST output the entire response (including animal type, disease names, reasoning, basic_response, and invalid_reason) in {lang_name} language.",
        "text_symptom_prompt_template": "You are an expert veterinary clinical symptom analyzer.\nThe user provided a clinical description of a sick animal without any image.\nAnimal Species: {animal_type} (Normalized: {norm_animal})\nDescription of observed signs and symptoms:\n{description}\n\nHere is the official reference list of 46 clinical symptoms:\n{symptoms_list_str}\n\nTasks:\n1. Identify which symptoms from the official reference list match the observed signs described in the text description.\n   Return their symptom numbers in `matched_symptom_numbers` (an empty list if none match).\n2. In `symptom_analysis`, provide a clear, supportive clinical assessment of the reported symptoms in {lang_name} language.\n3. For each matched symptom, look up its standard veterinary field handbook guidance and provide it in `symptom_guidance_entries` translated and written entirely in {lang_name} language.\n   Format each entry as: [Symptom Name in {lang_name}] Specific handbook guidance and practical field care steps in {lang_name} language.\n\nSTRICT CONSTRAINTS:\n- DO NOT name, diagnose, suspect, or identify any specific disease under any circumstances (e.g., do NOT mention FMD, Anthrax, Swine Fever, Rabies, etc.).\n- Focus strictly on the symptoms, clinical signs, and condition management.\n- All entries in `symptom_analysis` and `symptom_guidance_entries` MUST be written in {lang_name} language.\n- Return your response strictly matching the schema.",
        "cluster_trigger_diseases": [
            "Foot-and-mouth disease (FMD)",
            "African swine fever (ASF)",
            "Highly pathogenic avian influenza (HPAI)",
            "Newcastle disease",
            "Rabies",
            "Anthrax",
            "Lumpy skin disease (LSD)"
        ]
    }

app = FastAPI(
    title="FAO-PODD Animal Disease Diagnosis API",
    description="Analyze animal images and descriptions to identify potential diseases.",
    version="2.11.3"
)

EXCLUDED_LOG_PREFIXES = (
    "/api/logs",
    "/api/clusters",
    "/favicon.ico",
    "/robots.txt",
    "/static"
)
EXCLUDED_EXACT_GET_PATHS = (
    "/",
    "/index.html",
    "/privacy",
    "/home",
    "/about",
    "/landing"
)

@app.middleware("http")
async def universal_request_logging_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method

    # Skip internal dashboard polling and static assets
    if any(path.startswith(prefix) for prefix in EXCLUDED_LOG_PREFIXES):
        return await call_next(request)
    if method == "GET" and path in EXCLUDED_EXACT_GET_PATHS:
        return await call_next(request)

    start_time = time.time()
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    request.state.logged = False

    eff_tenant = (
        request.headers.get("x-ohtk-tenant")
        or request.headers.get("x-tenant-code")
    )
    req_params = {}
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            raw_body = await request.body()
            if raw_body:
                try:
                    req_params = json.loads(raw_body)
                    if isinstance(req_params, dict):
                        tenant_obj = req_params.get("tenant") or {}
                        if isinstance(tenant_obj, dict) and not eff_tenant:
                            eff_tenant = tenant_obj.get("schema") or tenant_obj.get("code")
                except Exception:
                    req_params = {"raw_body": raw_body[:1024].decode("utf-8", errors="ignore")}
        elif "multipart/form-data" in content_type:
            # Leave streaming multipart untouched for python-multipart
            req_params = dict(request.query_params)
            req_params["content_type"] = "multipart/form-data"
        else:
            if request.query_params:
                req_params = dict(request.query_params)
            if request.method in ("POST", "PUT", "PATCH"):
                raw_body = await request.body()
                if raw_body:
                    try:
                        req_params["raw"] = raw_body[:1024].decode("utf-8", errors="ignore")
                    except Exception:
                        pass
    except Exception as e:
        req_params = {"error_reading_request": str(e)}

    response = None
    try:
        response = await call_next(request)
    except Exception as exc:
        latency = time.time() - start_time
        if not getattr(request.state, "logged", False):
            log_request_response(
                method=method,
                path=path,
                status_code=500,
                latency=latency,
                client_ip=client_ip,
                request_params=req_params,
                response_body=json.dumps({"detail": f"Internal Server Error: {str(exc)}"}),
                tenant_key=eff_tenant or "default"
            )
            request.state.logged = True
        raise exc

    if getattr(request.state, "logged", False):
        return response

    latency = time.time() - start_time
    resp_body_bytes = b""
    try:
        async for chunk in response.body_iterator:
            resp_body_bytes += chunk
        new_response = Response(
            content=resp_body_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        resp_text = resp_body_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        new_response = response
        resp_text = f"Error reading response body: {str(e)}"

    log_request_response(
        method=method,
        path=path,
        status_code=response.status_code,
        latency=latency,
        client_ip=client_ip,
        request_params=req_params,
        response_body=resp_text,
        tenant_key=eff_tenant or "default"
    )
    request.state.logged = True
    return new_response

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
    if "prod.api.lahis.ohtk.org" in api_url:
        api_url = api_url.replace("prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org")
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
    if "prod.api.lahis.ohtk.org" in api_url:
        api_url = api_url.replace("prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org")
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
    if "prod.api.lahis.ohtk.org" in tenant_api_url:
        tenant_api_url = tenant_api_url.replace("prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org")
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

@app.get("/webhooks/lahis")
async def webhooks_lahis_health():
    return {"status": "ok", "message": "LAHIS webhook endpoint is active. Webhooks must be sent as HTTP POST with HMAC signature."}

@app.post("/analyze", response_model=DiagnosisResponse)
@app.post("/webhooks/lahis", response_model=DiagnosisResponse)
async def analyze_animal_image(
    request: Request,
    images: Optional[List[UploadFile]] = File(None, description="Images of the sick/diseased animal"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID to fetch the image from"),
    description: Optional[str] = Form(None, description="Optional text description of signs/symptoms"),
    animal_type: Optional[str] = Form(None, description="Animal type / species when submitting without image"),
    lang: str = Form("en", description="Language for response: 'en' (English), 'th' (Thai), 'lo' (Lao)"),
    tenant_code: Optional[str] = Form(None, description="Optional tenant code (e.g. 'local tester')")
):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    saved_image_path = None
    prompt_tokens = 0
    output_tokens = 0
    tenant_key = "default"
    tenant_ctx = None
    tenant_client = None
    job = None
    
    eff_tenant_code = (
        tenant_code
        or request.headers.get("x-tenant-code")
        or request.headers.get("x-ohtk-tenant")
    )
    if not eff_tenant_code and request.headers.get("x-source") == "web-tester":
        eff_tenant_code = "local tester"
    if eff_tenant_code:
        tenant_key = eff_tenant_code

    # Initialize request_params early to prevent UnboundLocalError in exception handler
    request_params = {
        "lang": lang,
        "description": description,
        "animal_type": animal_type,
        "report_id": report_id,
        "image_filenames": [img.filename for img in images] if images else [],
        "source": request.headers.get("x-source", "api"),
        "tenant": {
            "code": eff_tenant_code
        } if eff_tenant_code else None
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
        raw_body = b""
        
        if is_json:
            raw_body = await request.body()
            try:
                body = json.loads(raw_body)
                request_params = body
            except Exception:
                request_params = {"raw_body": raw_body.decode("utf-8", errors="ignore")}
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            if isinstance(body, dict):
                t_obj = body.get("tenant") or {}
                if isinstance(t_obj, dict):
                    t_cand = t_obj.get("schema") or t_obj.get("code")
                    if t_cand:
                        tenant_key = t_cand

            is_multi_tenant = bool(
                request.headers.get("x-ohtk-tenant")
                or request.headers.get("x-ohtk-signing-key-id")
                or request.headers.get("x-ohtk-signature-alg")
                or (isinstance(body.get("tenant"), dict) and body.get("tenant", {}).get("apiBaseUrl"))
            )

            if is_multi_tenant:
                tenant_ctx, body = await validate_webhook_request(
                    request=request,
                    raw_body=raw_body,
                    registry=tenant_registry,
                    expected_operation=["report.submitted", "ai.evaluation_requested"]
                )
                tenant_key = tenant_ctx.tenant_key
                tenant_client = LahisTenantClient(tenant_ctx, tenant_token_manager)
                api_url = tenant_ctx.api_base_url
                lang = tenant_ctx.lang
                event_type = body.get("eventType")
                event_id = body.get("eventId") or "unknown-event"
                report_data = body.get("report") or {}
                report_id = report_data.get("id")
                if not report_id:
                    raise HTTPException(status_code=400, detail="Missing report.id in payload")
                raw_desc = (
                    report_data.get("description")
                    or report_data.get("remarks")
                    or report_data.get("symptoms")
                    or report_data.get("symptom")
                    or report_data.get("notes")
                    or body.get("description")
                    or ""
                )
                event_info = f"Webhook Event ID: {event_id}. Report Type: {report_data.get('reportType', {}).get('name', 'Unknown')}"
                description = f"{raw_desc} ({event_info})" if raw_desc else event_info
                if not animal_type:
                    rep_animal = report_data.get("animalType") or report_data.get("animal_type") or body.get("animal_type")
                    if isinstance(rep_animal, dict):
                        animal_type = rep_animal.get("name") or rep_animal.get("name_en")
                    elif isinstance(rep_animal, str):
                        animal_type = rep_animal

                job, job_action = get_or_create_job(
                    tenant_key=tenant_key,
                    integration_code=tenant_ctx.integration_code,
                    operation=event_type or "report.submitted",
                    event_id=event_id,
                    raw_body_bytes=raw_body,
                    request_payload=body
                )
                if job_action == "replay":
                    if job.status == "COMPLETED" and job.model_result:
                        return DiagnosisResponse(**job.model_result)
                    elif job.status in ("ACCEPTED", "RUNNING", "CALLBACK_PENDING") and job.model_result:
                        return DiagnosisResponse(**job.model_result)
                update_job(job.job_key, status="RUNNING")
            else:
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
                if event_type not in ["report.submitted", "ai.evaluation_requested"]:
                    raise HTTPException(status_code=400, detail=f"Unsupported eventType: {event_type}")
                    
                report_data = body.get("report") or {}
                report_id = report_data.get("id")
                if not report_id:
                    raise HTTPException(status_code=400, detail="Missing report.id in payload")
                    
                event_id = body.get("eventId") or "unknown-event"
                lang = os.getenv("LAHIS_LANG", "lo")
                raw_desc = (
                    report_data.get("description")
                    or report_data.get("remarks")
                    or report_data.get("symptoms")
                    or report_data.get("symptom")
                    or report_data.get("notes")
                    or body.get("description")
                    or ""
                )
                event_info = f"Webhook Event ID: {event_id}. Report Type: {report_data.get('reportType', {}).get('name', 'Unknown')}"
                description = f"{raw_desc} ({event_info})" if raw_desc else event_info
                if not animal_type:
                    rep_animal = report_data.get("animalType") or report_data.get("animal_type") or body.get("animal_type")
                    if isinstance(rep_animal, dict):
                        animal_type = rep_animal.get("name") or rep_animal.get("name_en")
                    elif isinstance(rep_animal, str):
                        animal_type = rep_animal
        else:
            eff_tenant_code = tenant_code or request.headers.get("x-tenant-code")
            if not eff_tenant_code and request.headers.get("x-source") == "web-tester":
                eff_tenant_code = "local tester"
            if eff_tenant_code:
                tenant_key = eff_tenant_code
            request_params = {
                "lang": lang,
                "description": description,
                "animal_type": animal_type,
                "report_id": report_id,
                "image_filenames": [img.filename for img in images] if images else [],
                "source": request.headers.get("x-source", "api"),
                "tenant": {
                    "code": eff_tenant_code
                } if eff_tenant_code else None
            }
        steps["step_1"]["data"] = request_params
        print(f"[STEP 1] Request from LAHIS. Content-Type: {content_type}, is_json: {is_json}, params/body: {json.dumps(request_params)}")
        if lang not in ["en", "th", "lo"]:
            raise HTTPException(status_code=400, detail="Unsupported language. Supported languages are 'en', 'th', 'lo'.")
            
        if not is_json and not images and not report_id and not (description and animal_type):
            raise HTTPException(status_code=400, detail="Either image files or report_id must be provided. Alternatively, provide description and animal_type for text symptom consultation.")
        
        multiple_images_bytes = []
        api_url = tenant_ctx.api_base_url if tenant_ctx else os.getenv("TENANT_API_URL")
        client_id = tenant_ctx.oauth_client_id if tenant_ctx else os.getenv("LAHIS_CLIENT_ID")
        client_secret = tenant_ctx.oauth_client_secret if tenant_ctx else os.getenv("LAHIS_CLIENT_SECRET")
        token = None
        
        if images and not is_json:
            for img in images:
                try:
                    data = await img.read()
                    if data:
                        multiple_images_bytes.append(data)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid image file {img.filename}: {str(e)}")
        elif report_id:
            if tenant_client:
                multiple_images_bytes = await tenant_client.fetch_all_report_images_bytes(report_id)
            else:
                if not all([api_url, client_id, client_secret]):
                    raise HTTPException(
                        status_code=500,
                        detail="LAHIS integration environment variables (TENANT_API_URL, LAHIS_CLIENT_ID, LAHIS_CLIENT_SECRET) are not fully configured."
                    )
                token = await get_lahis_token(client_id, client_secret, api_url)
                multiple_images_bytes = await fetch_lahis_report_images(report_id, token, api_url)
            pass
            
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
                
                config = load_config()
                raw_text_prompt_template = config.get(
                    "text_symptom_prompt_template",
                    "You are an expert veterinary clinical symptom analyzer.\nThe user provided a clinical description of a sick animal without any image.\nAnimal Species: {animal_type} (Normalized: {norm_animal})\nDescription of observed signs and symptoms:\n{description}\n\nHere is the official reference list of 46 clinical symptoms:\n{symptoms_list_str}\n\nTasks:\n1. Identify which symptoms from the official reference list match the observed signs described in the text description.\n   Return their symptom numbers in `matched_symptom_numbers` (an empty list if none match).\n2. In `symptom_analysis`, provide a clear, supportive clinical assessment of the reported symptoms in {lang_name} language.\n3. For each matched symptom, look up its standard veterinary field handbook guidance and provide it in `symptom_guidance_entries` translated and written entirely in {lang_name} language.\n   Format each entry as: [Symptom Name in {lang_name}] Specific handbook guidance and practical field care steps in {lang_name} language.\n\nSTRICT CONSTRAINTS:\n- DO NOT name, diagnose, suspect, or identify any specific disease under any circumstances (e.g., do NOT mention FMD, Anthrax, Swine Fever, Rabies, etc.).\n- Focus strictly on the symptoms, clinical signs, and condition management.\n- All entries in `symptom_analysis` and `symptom_guidance_entries` MUST be written in {lang_name} language.\n- Return your response strictly matching the schema."
                )
                try:
                    text_prompt = raw_text_prompt_template.format(
                        animal_type=animal_type,
                        norm_animal=norm_animal,
                        description=description,
                        symptoms_list_str=symptoms_list_str,
                        lang_name=lang_name
                    )
                except Exception as e:
                    print(f"Error formatting text_symptom_prompt_template: {e}")
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
                "Analyze the provided image(s) and clinical text description.\n\nGuardrail Policy:\n{guardrail_policy}\n\nAllowed Priority Reportable Diseases List by Animal Species:\n{allowed_diseases}\n\nClinical Context & Reporter Information:\n- Text Description / Observed Clinical Signs: {description}\n- Stated Animal Species (if provided): {animal_species_context}\n\nMultimodal Diagnostic Guidelines:\n- MULTIMODAL SYNTHESIS: Perform a comprehensive multimodal veterinary evaluation by combining BOTH the visual evidence in the image(s) (e.g., visible lesions, blisters, vesicles, ulcers, erosions, discoloration, swellings, discharges, body posture, skin/mucosal changes) AND the clinical observations or history provided in the text description (e.g., high fever, sudden death, lameness, anorexia, salivation, diarrhea, coughing, abortion, progression rate, herd mortality).\n- The text description provides vital clinical context: use it alongside the image(s) to corroborate visual findings, differentiate diseases that share similar visual lesions, and assess overall disease severity.\n- If no text description is provided or if it states 'None provided', base your clinical evaluation solely on the visual findings in the image(s).\n\nTasks (Execute ONLY if `is_valid_animal_image` is True):\n1. Identify the type of animal (confirming against the stated animal species if provided).\n2. Multimodal Disease Evaluation: Evaluate the combined visual symptoms from the image(s) and clinical signs from the description ONLY against the allowed Priority Reportable Diseases list for that animal species. If the detected/suspected disease is NOT on this list, do NOT provide any disease suggestions (return `diseases: []`).\n3. If one or more diseases from the allowed list match the combined visual and described symptoms, list up to three with a confidence level (0.0 to 1.0), reasoning explaining how both the visual lesions and the described clinical signs support the diagnosis, and basic response recommendations for local veterinarians without requiring laboratory testing in {lang_name} language.\n4. If the confidence level is less than 0.8, you should not show the diseases suggestion, but ask the user to provide more information about the disease, perform additional lab testing, or gather more information.\n\nStrict Constraints:\n- NEVER diagnose, mention, or suggest any disease that is not explicitly on the allowed Priority Reportable Diseases list for that animal species. If not on the list, return `diseases: []`.\n- If the animal is not in the allowed list, `is_valid_animal_image` MUST be False and `invalid_reason` MUST state 'This is not an acceptable animal, we don't process [animal name]' in {lang_name} language.\n- You MUST output the entire response (including animal type, disease names, reasoning, basic_response, and invalid_reason) in {lang_name} language."
            )
            if "{allowed_diseases}" in prompt_template:
                prompt_template = prompt_template.replace("{allowed_diseases}", allowed_diseases_text)
            if "{allowed_animals}" in prompt_template:
                prompt_template = prompt_template.replace("{allowed_animals}", allowed_animals_text)

            animal_species_context = animal_type if animal_type else "Not specified (determine from image)"
            formatted_description = description if description and description.strip() else "None provided (analyze based purely on image)"

            try:
                prompt = prompt_template.format(
                    guardrail_policy=guardrail_policy,
                    lang_name=lang_name,
                    description=formatted_description,
                    animal_species_context=animal_species_context,
                    animal_type=animal_type or "Not specified"
                )
            except Exception as e:
                print(f"Error formatting prompt_template from config.json: {e}")
                prompt = (
                    f"Analyze the provided image(s) and clinical text description.\n\n"
                    f"Guardrail Policy:\n{guardrail_policy}\n\n"
                    f"Clinical Context & Reporter Information:\n"
                    f"- Text Description / Observed Clinical Signs: {formatted_description}\n"
                    f"- Stated Animal Species: {animal_species_context}\n\n"
                    f"Multimodal Diagnostic Guidelines:\n"
                    f"- Synthesize BOTH the visual evidence in the image(s) and the clinical observations in the text description.\n"
                    f"- Use the description directly alongside the image(s) to corroborate lesions and differentiate diseases.\n\n"
                    f"Tasks (Execute ONLY if `is_valid_animal_image` is True):\n"
                    f"1. Identify the type of animal.\n"
                    f"2. Multimodal Disease Evaluation: Evaluate the combined visual symptoms and described clinical signs against the allowed Priority Reportable Diseases list for that animal species.\n"
                    f"3. List up to three matching diseases with a confidence level (0.0 to 1.0), reasoning explaining how visual findings and described symptoms support the diagnosis, and basic response recommendations in {lang_name} language.\n\n"
                    f"Constraints:\n"
                    f"- You MUST output the entire response (including animal type, disease names, reasoning, and invalid_reason) in {lang_name} language."
                )
            
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
        
        if job:
            update_job(job.job_key, status="RESULT_READY", model_result=result)

        # If it was a webhook, post the feedback comment back to LAHIS
        if is_json:
            comment_body = format_diagnosis_comment(result, lang)
            # Find highest confidence score from diseases
            highest_confidence = 0.0
            for d in (result.get("diseases") or []):
                highest_confidence = max(highest_confidence, d.get("confidence", 0.0))

            comments_url = f"{api_url.rstrip('/')}/api/integrations/v1/reports/{report_id}/comments" if api_url else ""
            idempotency_key = f"ai-feedback-{event_id}"
            payload = {
                "externalActionId": idempotency_key,
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
            if job:
                update_job(job.job_key, status="CALLBACK_PENDING", callback_payload=payload)

            try:
                if tenant_client:
                    comment_response = await tenant_client.submit_comment(report_id, idempotency_key, payload)
                else:
                    if not token and all([client_id, client_secret, api_url]):
                        token = await get_lahis_token(client_id, client_secret, api_url)
                    comment_response = await submit_lahis_comment(report_id, event_id, comment_body, highest_confidence, api_url, token)

                if job:
                    update_job(job.job_key, status="COMPLETED", callback_response=comment_response)

                steps["step_5"] = {
                    "title": "5. Response back from LAHIS",
                    "data": comment_response
                }
            except HTTPException as e:
                if job:
                    update_job(job.job_key, status="FAILED", error_message=e.detail)
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
                    method=request.method,
                    path=request.url.path,
                    status_code=e.status_code,
                    latency=latency,
                    client_ip=client_ip,
                    request_params=request_params,
                    response_body=json.dumps({"detail": e.detail}),
                    steps_json=steps,
                    image_path=saved_image_path,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    tenant_key=tenant_key
                )
                request.state.logged = True
                raise e

        # Automated Priority Cluster Processing (Option A: Both /detect-cluster and post-/analyze)
        config = load_config()
        detected_priority_disease = None
        for d in (result.get("diseases") or []):
            if isinstance(d, dict) and d.get("name"):
                is_trig, canonical = is_cluster_trigger_disease(d["name"], config)
                if is_trig:
                    detected_priority_disease = canonical
                    break

        if detected_priority_disease and report_id:
            village_id = None
            village_name = None
            incident_loc = None
            incident_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            if is_json:
                report_data = body.get("report") or {}
                if report_data.get("incidentDate"):
                    incident_date = report_data.get("incidentDate")

            # Try to fetch fuller incident detail
            if tenant_client:
                try:
                    inc_detail = await tenant_client.fetch_incident(report_id)
                    inc_obj = inc_detail.get("incident") or inc_detail
                    v_info = inc_obj.get("village") or {}
                    village_id = v_info.get("id") or inc_obj.get("villageId") or inc_obj.get("village_id")
                    village_name = v_info.get("name") or inc_obj.get("villageName") or inc_obj.get("village_name")
                    incident_loc = inc_obj.get("location")
                    if inc_obj.get("incidentDate"):
                        incident_date = inc_obj.get("incidentDate")
                    if not village_id:
                        rel_auth = inc_obj.get("relevantAuthorityIds") or []
                        if rel_auth:
                            village_id = f"auth-{rel_auth[0]}"
                        elif incident_loc:
                            village_id = f"loc-{incident_loc.get('lat')},{incident_loc.get('lon')}"
                except Exception as e:
                    print(f"Error fetching incident detail for automated cluster creation: {e}")
            elif all([api_url, client_id, client_secret]):
                try:
                    if not token:
                        token = await get_lahis_token(client_id, client_secret, api_url)
                    inc_detail = await fetch_lahis_incident_detail(report_id, token, api_url)
                    inc_obj = inc_detail.get("incident") or inc_detail
                    v_info = inc_obj.get("village") or {}
                    village_id = v_info.get("id") or inc_obj.get("villageId") or inc_obj.get("village_id")
                    village_name = v_info.get("name") or inc_obj.get("villageName") or inc_obj.get("village_name")
                    incident_loc = inc_obj.get("location")
                    if inc_obj.get("incidentDate"):
                        incident_date = inc_obj.get("incidentDate")
                    if not village_id:
                        rel_auth = inc_obj.get("relevantAuthorityIds") or []
                        if rel_auth:
                            village_id = f"auth-{rel_auth[0]}"
                        elif incident_loc:
                            village_id = f"loc-{incident_loc.get('lat')},{incident_loc.get('lon')}"
                except Exception as e:
                    print(f"Error fetching incident detail for automated cluster creation: {e}")

            if not village_id:
                if is_json:
                    rel_auth = (body.get("report") or {}).get("relevantAuthorityIds") or []
                    village_id = f"auth-{rel_auth[0]}" if rel_auth else "default_village"
                else:
                    village_id = "default_village"

            # 1. Record cluster report in DB with detected disease and tenant_key
            record_cluster_report(
                report_id=str(report_id),
                village_id=str(village_id),
                village_name=village_name,
                date_str=incident_date,
                raw_payload=json.dumps(request_params),
                detected_disease=detected_priority_disease,
                tenant_key=tenant_key
            )

            # 2. Automatically submit cluster to LAHIS if incoming webhook
            if is_json and (tenant_client or (all([api_url, client_id, client_secret]) and token)):
                all_village_reports = get_village_all_reports(str(village_id), tenant_key=tenant_key)
                all_incident_ids = list(set([r["report_id"] for r in all_village_reports if r.get("report_id")]))
                if report_id not in all_incident_ids:
                    all_incident_ids.append(report_id)

                village_id_int = int(village_id) if str(village_id).isdigit() else None
                authority_ids = (body.get("report") or {}).get("relevantAuthorityIds") if is_json else []

                schema_slug = tenant_ctx.schema if tenant_ctx else "legacy"
                cluster_external_id = f"cluster-{schema_slug}-{village_id}-{incident_date}"

                cluster_payload = {
                    "externalClusterId": cluster_external_id,
                    "algorithmVersion": "podd-cluster-detector-v2",
                    "window": {
                        "from": incident_date,
                        "to": incident_date
                    },
                    "incidentIds": all_incident_ids,
                    "authorityIds": authority_ids or [],
                    "villageIds": [village_id_int] if village_id_int is not None else [],
                    "geometry": incident_loc,
                    "radiusMeters": 500.0,
                    "score": 1.0,
                    "riskLevel": "HIGH",
                    "explanation": f"Cluster detected: Priority reportable disease '{detected_priority_disease}' detected by AI in village {village_name or village_id}.",
                    "metadata": {
                        "source": "fao-podd-ai-analyzer",
                        "detected_disease": detected_priority_disease,
                        "report_count": len(all_incident_ids),
                        "tenant_key": tenant_key
                    }
                }
                try:
                    if tenant_client:
                        cluster_sub_resp = await tenant_client.submit_cluster(cluster_external_id, cluster_payload)
                    else:
                        cluster_sub_resp = await submit_lahis_cluster(cluster_payload, token, api_url)
                    steps["step_6"] = {
                        "title": "6. Automated Priority Cluster Submission to LAHIS",
                        "data": {
                            "payload": cluster_payload,
                            "response": cluster_sub_resp
                        }
                    }
                except Exception as e:
                    print(f"Failed to auto-submit priority cluster to LAHIS in /analyze: {e}")
                    steps["step_6"] = {
                        "title": "6. Automated Priority Cluster Submission to LAHIS (Error)",
                        "data": {"error": str(e)}
                    }
        
        # Log success
        latency = time.time() - start_time
        log_request_response(
            method=request.method,
            path=request.url.path,
            status_code=200,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps(result),
            steps_json=steps,
            image_path=saved_image_path,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tenant_key=tenant_key
        )
        request.state.logged = True
        return result

    except HTTPException as e:
        latency = time.time() - start_time
        log_request_response(
            method=request.method,
            path=request.url.path,
            status_code=e.status_code,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": e.detail}),
            steps_json=steps,
            image_path=saved_image_path,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tenant_key=tenant_key
        )
        request.state.logged = True
        raise e
    except Exception as e:
        latency = time.time() - start_time
        log_request_response(
            method=request.method,
            path=request.url.path,
            status_code=500,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": str(e)}),
            steps_json=steps,
            image_path=saved_image_path,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            tenant_key=tenant_key
        )
        request.state.logged = True
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

@app.post("/detect-cluster")
async def detect_cluster(
    request: Request,
    images: Optional[List[UploadFile]] = File(None, description="Optional images"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID"),
    description: Optional[str] = Form(None, description="Optional text description"),
    lang: str = Form("en", description="Language for response: 'en', 'th', 'lo'"),
    tenant_code: Optional[str] = Form(None, description="Optional tenant code (e.g. 'local tester')")
):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    tenant_key = "default"
    tenant_ctx = None
    tenant_client = None
    job = None
    
    eff_tenant_code = tenant_code or request.headers.get("x-tenant-code")
    if not eff_tenant_code and request.headers.get("x-source") == "web-tester":
        eff_tenant_code = "local tester"
    if eff_tenant_code:
        tenant_key = eff_tenant_code

    request_params = {
        "lang": lang,
        "description": description,
        "report_id": report_id,
        "image_filenames": [img.filename for img in images] if images else [],
        "source": request.headers.get("x-source", "api"),
        "tenant": {
            "code": eff_tenant_code
        } if eff_tenant_code else None
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

            is_multi_tenant = bool(
                request.headers.get("x-ohtk-tenant")
                or request.headers.get("x-ohtk-signing-key-id")
                or request.headers.get("x-ohtk-signature-alg")
                or (isinstance(body.get("tenant"), dict) and body.get("tenant", {}).get("apiBaseUrl"))
            )

            if is_multi_tenant:
                tenant_ctx, body = await validate_webhook_request(
                    request=request,
                    raw_body=raw_body_bytes,
                    registry=tenant_registry,
                    expected_operation=("report.submitted", "detect-cluster")
                )
                tenant_key = tenant_ctx.tenant_key
                tenant_client = LahisTenantClient(tenant_ctx, tenant_token_manager)
                api_url = tenant_ctx.api_base_url
                event_id = body.get("eventId") or "unknown-event"
                report_data = body.get("report") or {}
                report_id = report_data.get("id")
                if not report_id:
                    raise HTTPException(status_code=400, detail="Missing report.id in payload")

                job, job_action = get_or_create_job(
                    tenant_key=tenant_key,
                    integration_code=tenant_ctx.integration_code,
                    operation="detect-cluster",
                    event_id=event_id,
                    raw_body_bytes=raw_body_bytes,
                    request_payload=body
                )
                if job_action == "replay":
                    if job.status == "COMPLETED" and job.model_result:
                        return job.model_result
                    elif job.status in ("ACCEPTED", "RUNNING", "CALLBACK_PENDING") and job.model_result:
                        return job.model_result
                update_job(job.job_key, status="RUNNING")
            else:
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
            
        api_url = tenant_ctx.api_base_url if tenant_ctx else os.getenv("TENANT_API_URL")
        client_id = tenant_ctx.oauth_client_id if tenant_ctx else os.getenv("LAHIS_CLIENT_ID")
        client_secret = tenant_ctx.oauth_client_secret if tenant_ctx else os.getenv("LAHIS_CLIENT_SECRET")
        token = None
        
        # Step 2: Fetch incident detail from LAHIS
        if tenant_client:
            incident_data = await tenant_client.fetch_incident(report_id)
        else:
            if not all([api_url, client_id, client_secret]):
                raise HTTPException(
                    status_code=500,
                    detail="LAHIS integration environment variables (TENANT_API_URL, LAHIS_CLIENT_ID, LAHIS_CLIENT_SECRET) are not fully configured."
                )
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
                
        incident_date = incident.get("incidentDate") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Load config to get trigger diseases list
        config = load_config()

        # Step A: Collect candidate diseases from all available sources
        candidate_diseases = []

        # 1. Direct from request params / form
        for k in ["disease", "detected_disease", "disease_name"]:
            val = request_params.get(k)
            if val and isinstance(val, str):
                candidate_diseases.append(val)

        # 2. From incident details (LAHIS)
        for k in ["disease", "detected_disease", "diseaseName"]:
            val = incident.get(k)
            if val and isinstance(val, str):
                candidate_diseases.append(val)
        if incident.get("diseases") and isinstance(incident.get("diseases"), list):
            for d in incident["diseases"]:
                if isinstance(d, str):
                    candidate_diseases.append(d)
                elif isinstance(d, dict) and d.get("name"):
                    candidate_diseases.append(d["name"])

        # 3. From AI diagnosis logs in DB for this report_id (scoped to tenant)
        ai_logged_diseases = get_ai_diagnosed_diseases_for_report(str(report_id), tenant_key=tenant_key)
        candidate_diseases.extend(ai_logged_diseases)

        # 4. If images provided with detect-cluster or in LAHIS and no candidate disease found yet
        if not candidate_diseases:
            multiple_images_bytes = []
            if images:
                for img in images:
                    try:
                        d = await img.read()
                        if d:
                            multiple_images_bytes.append(d)
                    except Exception:
                        pass
            elif tenant_client:
                try:
                    multiple_images_bytes = await tenant_client.fetch_all_report_images_bytes(report_id)
                except Exception as e:
                    print(f"Could not fetch LAHIS images for cluster detection: {e}")
            elif all([api_url, client_id, client_secret]) and report_id:
                try:
                    if not token:
                        token = await get_lahis_token(client_id, client_secret, api_url)
                    multiple_images_bytes = await fetch_lahis_report_images(report_id, token, api_url)
                except Exception as e:
                    print(f"Could not fetch LAHIS images for cluster detection: {e}")

            if multiple_images_bytes:
                try:
                    genai_client = get_genai_client()
                    pil_imgs = [Image.open(io.BytesIO(b)) for b in multiple_images_bytes]
                    ai_resp = genai_client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=pil_imgs + ["Identify animal species and any symptoms or diseases matching the priority reportable diseases list."],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=DiagnosisResponse,
                            temperature=0.2,
                        ),
                    )
                    parsed_ai = json.loads(ai_resp.text)
                    for d in (parsed_ai.get("diseases") or []):
                        if isinstance(d, dict) and d.get("name"):
                            candidate_diseases.append(d["name"])
                except Exception as e:
                    print(f"Error running on-demand AI analysis in cluster detector: {e}")

        # Check candidate diseases against priority trigger list
        matched_trigger_disease = None
        for cd in candidate_diseases:
            is_trig, trig_canonical = is_cluster_trigger_disease(cd, config)
            if is_trig:
                matched_trigger_disease = trig_canonical
                break

        # Record this incoming request to DB with detected disease and tenant_key
        record_cluster_report(
            report_id=report_id,
            village_id=str(village_id),
            village_name=village_name,
            date_str=incident_date,
            raw_payload=json.dumps(request_params),
            detected_disease=matched_trigger_disease or (candidate_diseases[0] if candidate_diseases else ""),
            tenant_key=tenant_key
        )

        # Check all reports for this village across history (tenant scoped)
        all_village_reports = get_village_all_reports(str(village_id), tenant_key=tenant_key)
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
            earliest_date = datetime.now(timezone.utc)
            latest_date = datetime.now(timezone.utc)
            span_days = 0

        # If not matched on current report, check if any prior report in village had a trigger disease
        if not matched_trigger_disease:
            for r in all_village_reports:
                r_dis = r.get("detected_disease")
                if r_dis:
                    is_trig, trig_canonical = is_cluster_trigger_disease(r_dis, config)
                    if is_trig:
                        matched_trigger_disease = trig_canonical
                        break

        cluster_detected = bool(matched_trigger_disease)
        cluster_response = None

        steps["step_3"] = {
            "title": "3. Priority Disease Cluster Check",
            "data": {
                "village_id": str(village_id),
                "village_name": village_name,
                "earliest_date": earliest_date.strftime("%Y-%m-%d"),
                "latest_date": latest_date.strftime("%Y-%m-%d"),
                "span_days": span_days,
                "report_count": report_count,
                "detected_disease": matched_trigger_disease,
                "cluster_detected": cluster_detected,
                "trigger_diseases": config.get("cluster_trigger_diseases", [])
            }
        }

        if cluster_detected:
            # Collect unique incident IDs in this village
            all_incident_ids = list(set([r["report_id"] for r in all_village_reports if r.get("report_id")]))
            if report_id not in all_incident_ids:
                all_incident_ids.append(report_id)

            village_id_int = int(village_id) if str(village_id).isdigit() else None
            authority_ids = incident.get("relevantAuthorityIds") or []

            schema_slug = tenant_ctx.schema if tenant_ctx else "legacy"
            cluster_external_id = f"cluster-{schema_slug}-{village_id}-{earliest_date.strftime('%Y%m%d')}-{latest_date.strftime('%Y%m%d')}"

            cluster_payload = {
                "externalClusterId": cluster_external_id,
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
                "explanation": f"Cluster detected: Priority reportable disease '{matched_trigger_disease}' detected by AI in village {village_name or village_id}.",
                "metadata": {
                    "source": "fao-podd-cluster-detector",
                    "detected_disease": matched_trigger_disease,
                    "span_days": span_days,
                    "report_count": report_count,
                    "tenant_key": tenant_key
                }
            }

            steps["step_4"] = {
                "title": "4. Sending Cluster to LAHIS Callback",
                "data": cluster_payload
            }

            if job:
                update_job(job.job_key, status="CALLBACK_PENDING", callback_payload=cluster_payload)

            try:
                if tenant_client:
                    cluster_response = await tenant_client.submit_cluster(cluster_external_id, cluster_payload)
                else:
                    if not token and all([client_id, client_secret, api_url]):
                        token = await get_lahis_token(client_id, client_secret, api_url)
                    cluster_response = await submit_lahis_cluster(cluster_payload, token, api_url)
                
                if job:
                    update_job(job.job_key, status="COMPLETED", callback_response=cluster_response)

                steps["step_5"] = {
                    "title": "5. LAHIS Callback Response",
                    "data": cluster_response
                }
            except HTTPException as e:
                if job:
                    update_job(job.job_key, status="FAILED", error_message=e.detail)
                steps["step_5"] = {
                    "title": "5. LAHIS Callback Response (Error)",
                    "data": {"status_code": e.status_code, "detail": e.detail}
                }
                latency = time.time() - start_time
                log_request_response(
                    method=request.method,
                    path=request.url.path,
                    status_code=e.status_code,
                    latency=latency,
                    client_ip=client_ip,
                    request_params=request_params,
                    response_body=json.dumps({"detail": e.detail}),
                    steps_json=steps,
                    tenant_key=tenant_key
                )
                request.state.logged = True
                # Propagate downstream callback failure
                raise e
                
        result = {
            "status": "success",
            "cluster_detected": cluster_detected,
            "detected_disease": matched_trigger_disease,
            "report_count": report_count,
            "village_id": str(village_id),
            "village_name": village_name,
            "earliest_date": earliest_date.strftime("%Y-%m-%d"),
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "span_days": span_days,
            "cluster_response": cluster_response
        }
        if job:
            update_job(job.job_key, status="COMPLETED", model_result=result, callback_response=cluster_response)
        
        latency = time.time() - start_time
        log_request_response(
            method=request.method,
            path=request.url.path,
            status_code=200,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps(result),
            steps_json=steps,
            tenant_key=tenant_key
        )
        request.state.logged = True
        return result

    except HTTPException as e:
        latency = time.time() - start_time
        log_request_response(
            method=request.method,
            path=request.url.path,
            status_code=e.status_code,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": e.detail}),
            steps_json=steps,
            tenant_key=tenant_key
        )
        request.state.logged = True
        raise e
    except Exception as e:
        latency = time.time() - start_time
        log_request_response(
            method=request.method,
            path=request.url.path,
            status_code=500,
            latency=latency,
            client_ip=client_ip,
            request_params=request_params,
            response_body=json.dumps({"detail": str(e)}),
            steps_json=steps,
            tenant_key=tenant_key
        )
        request.state.logged = True
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



def fetch_all_logs_data(limit: Optional[int] = 50, tenant: Optional[str] = None):
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ApiLog")
            if tenant:
                query.add_filter("tenant_key", "=", str(tenant))
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

                t_key = entity.get("tenant_key", "standalone")
                t_code = None
                if isinstance(parsed_params, dict) and isinstance(parsed_params.get("tenant"), dict):
                    t_code = parsed_params["tenant"].get("code") or parsed_params["tenant"].get("schema")
                if not t_code:
                    t_code = t_key.split(":")[-1] if (t_key and ":" in t_key) else (t_key or "-")

                logs.append({
                    "id": entity.key.id,
                    "tenant_key": t_key,
                    "tenant_id": t_key,
                    "tenant_code": t_code,
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
        if tenant:
            if limit and limit > 0:
                cursor.execute("SELECT * FROM api_logs WHERE tenant_key = ? ORDER BY id DESC LIMIT ?", (str(tenant), limit))
            else:
                cursor.execute("SELECT * FROM api_logs WHERE tenant_key = ? ORDER BY id DESC", (str(tenant),))
        else:
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

            t_key = row["tenant_key"] if "tenant_key" in row.keys() else "standalone"
            t_code = None
            if isinstance(parsed_params, dict) and isinstance(parsed_params.get("tenant"), dict):
                t_code = parsed_params["tenant"].get("code") or parsed_params["tenant"].get("schema")
            if not t_code:
                t_code = t_key.split(":")[-1] if (t_key and ":" in t_key) else (t_key or "-")

            logs.append({
                "id": row["id"],
                "tenant_key": t_key,
                "tenant_id": t_key,
                "tenant_code": t_code,
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
async def get_logs(limit: Optional[int] = 50, tenant: Optional[str] = None, username: str = Depends(authenticate_dashboard)):
    return fetch_all_logs_data(limit=limit, tenant=tenant)


@app.get("/api/logs/export")
async def export_logs(format: str = "json", tenant: Optional[str] = None, username: str = Depends(authenticate_dashboard)):
    logs = fetch_all_logs_data(limit=None, tenant=tenant)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format.lower() == "csv":
        output = io.StringIO()
        fieldnames = [
            "id", "tenant_code", "tenant_key", "timestamp", "method", "path", "status_code", "latency",
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
                elif req_params.get("eventType"):
                    target = f"Event: {req_params['eventType']}"

            if target == "None":
                status = log.get("status_code", 200)
                if status and status >= 400:
                    resp_body = log.get("response_body") or ""
                    try:
                        parsed = json.loads(resp_body)
                        detail = parsed.get("detail") or parsed.get("error")
                        target = f"Error: {detail}" if detail else f"HTTP {status}"
                    except Exception:
                        target = f"HTTP {status}"
                elif log.get("path") == "/webhooks/lahis" and log.get("method") == "GET":
                    target = "Webhook Health Probe"
                elif log.get("path"):
                    target = str(log.get("path"))

            writer.writerow({
                "id": log.get("id"),
                "tenant_code": log.get("tenant_code") or log.get("tenant_key", "default"),
                "tenant_key": log.get("tenant_key", "default"),
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
async def get_clusters(limit: int = 100, tenant: Optional[str] = None, username: str = Depends(authenticate_dashboard)):
    reports = []
    if USE_DATASTORE and datastore_client:
        try:
            query = datastore_client.query(kind="ClusterReport")
            if tenant:
                query.add_filter("tenant_key", "=", str(tenant))
            query.order = ["-timestamp"]
            results = list(query.fetch(limit=limit))
            for entity in results:
                reports.append({
                    "id": entity.key.id,
                    "tenant_key": entity.get("tenant_key", "default"),
                    "timestamp": entity.get("timestamp"),
                    "date_str": entity.get("date_str"),
                    "report_id": entity.get("report_id"),
                    "village_id": entity.get("village_id"),
                    "village_name": entity.get("village_name"),
                    "raw_payload": entity.get("raw_payload"),
                    "detected_disease": entity.get("detected_disease", "")
                })
        except Exception as e:
            print(f"Datastore fetch cluster reports error: {e}")

    if not reports:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if tenant:
                cursor.execute("SELECT * FROM cluster_reports WHERE tenant_key = ? ORDER BY id DESC LIMIT ?", (str(tenant), limit))
            else:
                cursor.execute("SELECT * FROM cluster_reports ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            conn.close()
            for row in rows:
                reports.append({
                    "id": row["id"],
                    "tenant_key": row["tenant_key"] if "tenant_key" in row.keys() else "default",
                    "timestamp": row["timestamp"],
                    "date_str": row["date_str"],
                    "report_id": row["report_id"],
                    "village_id": row["village_id"],
                    "village_name": row["village_name"],
                    "raw_payload": row["raw_payload"],
                    "detected_disease": row["detected_disease"] if "detected_disease" in row.keys() else ""
                })
        except Exception as e:
            print(f"SQLite fetch cluster reports error: {e}")

    # Group by (tenant_key, village_id) so villages from different tenants never collide
    groups = {}
    for r in reports:
        t_key = r.get("tenant_key") or "default"
        vid = r["village_id"]
        group_key = f"{t_key}:{vid}"
        if group_key not in groups:
            groups[group_key] = {
                "group_key": group_key,
                "tenant_key": t_key,
                "village_id": vid,
                "village_name": r["village_name"] or "Unknown Village",
                "reports": [],
                "dates": [],
                "detected_diseases": [],
                "latest_timestamp": r["timestamp"]
            }
        if r["report_id"] and r["report_id"] not in groups[group_key]["reports"]:
            groups[group_key]["reports"].append(r["report_id"])

        if r.get("detected_disease") and r["detected_disease"] not in groups[group_key]["detected_diseases"]:
            groups[group_key]["detected_diseases"].append(r["detected_disease"])

        d_str = r.get("date_str")
        if d_str:
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                groups[group_key]["dates"].append(dt)
            except Exception:
                pass

    config = load_config()
    grouped_list = []
    total_clusters = 0

    for g_key, g in groups.items():
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

        matched_diseases = []
        for dis in g.get("detected_diseases", []):
            if dis:
                is_trig, canonical = is_cluster_trigger_disease(dis, config)
                if is_trig and canonical not in matched_diseases:
                    matched_diseases.append(canonical)

        is_cluster = len(matched_diseases) > 0
        if is_cluster:
            total_clusters += 1

        grouped_list.append({
            "group_key": g["group_key"],
            "tenant_key": g["tenant_key"],
            "village_id": g["village_id"],
            "village_name": g["village_name"],
            "start_date": start_date_str,
            "latest_date": latest_date_str,
            "span_days": span_days,
            "report_count": len(g["reports"]),
            "reports": g["reports"],
            "cluster_detected": is_cluster,
            "trigger_disease": ", ".join(matched_diseases) if matched_diseases else None,
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

