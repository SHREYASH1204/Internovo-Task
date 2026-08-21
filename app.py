import os
import json
import csv
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

CSV_FILE = "cleaned_listings.csv"
CSV_HEADERS = ["bhk", "property_type", "locality", "area_sqft"]

# Initialize CSV file strictly with only the 4 required fields
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    else:
        # If header doesn't match exact required columns, recreate with proper headers
        try:
            with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header != CSV_HEADERS:
                with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)
        except Exception:
            with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

init_csv()

def save_to_csv(data: dict):
    """
    Saves exactly: bhk (number or null), property_type (flat/villa/plot/other or null),
    locality (string or null), and area_sqft (number or null). No other fields.
    """
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            data.get("bhk") if data.get("bhk") is not None else "",
            data.get("property_type") if data.get("property_type") is not None else "",
            data.get("locality") if data.get("locality") is not None else "",
            data.get("area_sqft") if data.get("area_sqft") is not None else "",
        ])

app = FastAPI(
    title="propOG Property Listing Cleaner API",
    description="AI service that cleans rushed listing descriptions, extracts structured property data, and saves to CSV.",
    version="1.0.0"
)

# Enable CORS for frontend integration flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input request schema
class ListingCleanRequest(BaseModel):
    raw_description: str = Field(..., json_schema_extra={"example": "2bhk flat for sale in koramangala 1200 sqft urgent sale prime loc near metro"})

# Schema used for LLM response validation & structuring
class PropertyStructuredData(BaseModel):
    headline: str = Field(description="A clean, catchy headline for the property listing.")
    description: str = Field(description="A polished, professional short description fixing typos and grammar.")
    tags: List[str] = Field(description="3 to 5 relevant tags summarizing key facts from the text.")
    bhk: Optional[float] = Field(default=None, description="Number of BHK / bedrooms mentioned, or null if not explicitly mentioned.")
    property_type: Optional[Literal["flat", "villa", "plot", "other"]] = Field(default=None, description="Property type ('flat', 'villa', 'plot', 'other'), or null if not explicitly mentioned.")
    locality: Optional[str] = Field(default=None, description="Locality, neighborhood, or city name mentioned, or null if not mentioned.")
    area_sqft: Optional[float] = Field(default=None, description="Built up or carpet area in sqft (numeric value), or null if area in sqft is not mentioned.")

# Full response schema returned by endpoint
class ListingCleanResponse(BaseModel):
    headline: str
    description: str
    tags: List[str]
    bhk: Optional[float] = None
    property_type: Optional[Literal["flat", "villa", "plot", "other"]] = None
    locality: Optional[str] = None
    area_sqft: Optional[float] = None
    missing_fields: List[str]
    needs_more_info: bool = False
    csv_saved: bool = True

SYSTEM_INSTRUCTION = """
You are an expert real estate data assistant for propOG.
Your task is to take a raw, rushed, typo-ridden property note written by an agent, clean it up into a professional listing, and extract structured metadata.

STRICT HARD RULES (NO HALLUCINATION):
1. DO NOT INVENT, GUESS, OR ASSUME ANY VALUES THAT ARE NOT EXPLICITLY MENTIONED OR DIRECTLY INFERRED FROM THE INPUT TEXT.
2. If BHK (number of bedrooms) is not mentioned in the input -> bhk MUST be null.
3. If property_type (flat, villa, plot, other) is not mentioned in the input -> property_type MUST be null.
4. If locality / location / neighborhood is not mentioned in the input -> locality MUST be null.
5. If area in sqft is not mentioned in the input -> area_sqft MUST be null.
6. Provide a headline, a short cleaned description, and 3-5 concise tags based strictly on facts present in the raw input note.
"""

def _call_single_ai_attempt(raw_description: str) -> PropertyStructuredData:
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        # Strip potential surrounding quotes from .env
        gemini_key = gemini_key.strip().strip('"').strip("'")
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=gemini_key)
        prompt = f"Raw property note:\n{raw_description}"
        
        # Primary and fallback Gemini models in case of rate limits/quota on free tier
        # Primary and fallback Gemini models
        models_to_try = [
            os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash"
        ]
        # Keep unique models in order
        models_to_try = list(dict.fromkeys(models_to_try))
        
        last_error = None
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=PropertyStructuredData,
                        temperature=0.1,
                    )
                )
                
                raw_json = response.text
                if not raw_json:
                    raise ValueError("Empty response received from Gemini model.")
                    
                data = json.loads(raw_json)
                if not isinstance(data, dict):
                    raise ValueError("Parsed JSON is not an object.")
                    
                # Pydantic validation of keys and types
                validated_obj = PropertyStructuredData(**data)
                return validated_obj
            except Exception as e:
                err_str = str(e)
                last_error = e
                # If 404 Not Found, 429 Resource Exhausted, or rate/quota limit, try next model
                if any(x in err_str for x in ["404", "NOT_FOUND", "no longer available", "429", "RESOURCE_EXHAUSTED", "quota"]):
                    print(f"[Warning] Model {model_name} failed ({err_str[:120]}...). Trying fallback model...")
                    continue
                else:
                    raise e
        
        if last_error:
            raise last_error
        
    elif openai_key:
        import openai
        openai_key = openai_key.strip().strip('"').strip("'")
        client = openai.OpenAI(api_key=openai_key)
        
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Raw property note:\n{raw_description}"}
            ],
            response_format=PropertyStructuredData,
            temperature=0.1
        )
        validated_obj = response.choices[0].message.parsed
        if validated_obj is None:
            raise ValueError("OpenAI returned empty structured response.")
        return validated_obj
    else:
        raise HTTPException(
            status_code=500,
            detail="No AI API key found. Please configure GEMINI_API_KEY or OPENAI_API_KEY in environment or .env file."
        )

def call_ai_api_with_retry(raw_description: str) -> PropertyStructuredData:
    """
    Calls AI API and validates JSON structure & types.
    Retries once if JSON parsing or type validation fails.
    """
    try:
        return _call_single_ai_attempt(raw_description)
    except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as err:
        print(f"[Warning] AI response validation failed on 1st attempt: {err}. Retrying once...")
        try:
            return _call_single_ai_attempt(raw_description)
        except Exception as retry_err:
            raise ValueError(f"AI response validation failed after retry: {str(retry_err)}")

@app.post("/api/clean-listing", response_model=ListingCleanResponse)
def clean_listing(payload: ListingCleanRequest):
    raw_text = payload.raw_description.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_description cannot be empty.")
    
    try:
        extracted = call_ai_api_with_retry(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API processing error: {str(e)}")
    
    # Identify which of the 4 structured keys came back null
    structured_keys = ["bhk", "property_type", "locality", "area_sqft"]
    missing_fields = []
    
    extracted_dict = extracted.model_dump()
    for key in structured_keys:
        if extracted_dict.get(key) is None:
            missing_fields.append(key)
            
    # Flag as "needs more info" if more than two fields (3 or 4) came back missing
    needs_more_info = len(missing_fields) > 2
            
    response_data = ListingCleanResponse(
        headline=extracted.headline,
        description=extracted.description,
        tags=extracted.tags,
        bhk=extracted.bhk,
        property_type=extracted.property_type,
        locality=extracted.locality,
        area_sqft=extracted.area_sqft,
        missing_fields=missing_fields,
        needs_more_info=needs_more_info,
        csv_saved=True
    )
    
    # Save strictly the 4 fields to CSV file: bhk, property_type, locality, area_sqft
    save_to_csv(response_data.model_dump())
    
    return response_data

@app.get("/api/download-csv")
def download_csv():
    if os.path.exists(CSV_FILE):
        return FileResponse(CSV_FILE, filename="cleaned_listings.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="CSV file not found.")

# Mount static folder and serve frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    print("Starting propOG Listing Cleaner server on http://127.0.0.1:8005 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8005, reload=True)
