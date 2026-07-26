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

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_animal_image(
    image: UploadFile = File(..., description="Image of the sick/diseased animal"),
    description: Optional[str] = Form(None, description="Optional text description of signs/symptoms"),
    lang: str = Form("en", description="Language for response: 'en' (English), 'th' (Thai), 'lo' (Lao)")
):
    if lang not in ["en", "th", "lo"]:
        raise HTTPException(status_code=400, detail="Unsupported language. Supported languages are 'en', 'th', 'lo'.")
    
    # Read image bytes
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
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
