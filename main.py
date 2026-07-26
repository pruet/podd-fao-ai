import os
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from PIL import Image
import json
import httpx
from dotenv import load_dotenv


load_dotenv()

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
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
        token_url = f"{api_url.rstrip('/')}/oauth/token/"
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
        report_url = f"{api_url.rstrip('/')}/api/v1/reports/{report_id}/"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = await client.get(report_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            report_data = response.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch report {report_id} from LAHIS: {str(e)}"
            )
        
        image_url = None
        # 1. Check for 'images' list
        if "images" in report_data and isinstance(report_data["images"], list) and len(report_data["images"]) > 0:
            first_img = report_data["images"][0]
            if isinstance(first_img, dict):
                image_url = first_img.get("url") or first_img.get("file") or first_img.get("image")
            elif isinstance(first_img, str):
                image_url = first_img
        # 2. Check direct fields
        if not image_url:
            image_url = report_data.get("image_url") or report_data.get("image") or report_data.get("file")
            
        if not image_url:
            raise HTTPException(
                status_code=404,
                detail=f"No animal image found in LAHIS report {report_id}."
            )
            
        try:
            img_headers = {}
            if image_url.startswith(api_url):
                img_headers["Authorization"] = f"Bearer {token}"
            img_response = await client.get(image_url, headers=img_headers, timeout=20.0)
            img_response.raise_for_status()
            return img_response.content
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download image from {image_url}: {str(e)}"
            )

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_animal_image(
    image: Optional[UploadFile] = File(None, description="Image of the sick/diseased animal"),
    report_id: Optional[str] = Form(None, description="LAHIS Report ID to fetch the image from"),
    description: Optional[str] = Form(None, description="Optional text description of signs/symptoms"),
    lang: str = Form("en", description="Language for response: 'en' (English), 'th' (Thai), 'lo' (Lao)")
):
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
    
    try:
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
        import json
        result = json.loads(response.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port)
