from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict
from recommender.recommend import recommend_vpn
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VPN Recommender API")

# Configure templates
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files if needed
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class VPNRecommendationRequest(BaseModel):
    speed: float = Field(..., gt=0, le=1000, description="Speed in Mbps (1-1000)")
    price: float = Field(..., gt=0, le=100, description="Monthly price in USD (0-100)")
    max_devices: int = Field(..., gt=0, le=100, description="Maximum allowed devices (1-100)")
    logging_policy: str = Field(..., pattern="^(no_logs|partial_logs)$", description="Logging policy preference")
    encryption: str = Field(..., description="Preferred encryption method")
    trial_available: str = Field(..., pattern="^(yes|no)$", description="Trial availability required")
    country: str = Field(..., min_length=2, description="Preferred country location")

    @validator('encryption')
    def validate_encryption(cls, v):
        valid_encryptions = [
            'AES-256', 'AES-128', 'ChaCha20', 
            'Blowfish-128', 'Blowfish-256',
            'RSA-2048', 'RSA-4096', 'SHA', 'MPPE'
        ]
        if v not in valid_encryptions:
            raise ValueError(f"Invalid encryption. Must be one of: {', '.join(valid_encryptions)}")
        return v

@app.on_event("startup")
async def startup_event():
    """Initialize application resources"""
    logger.info("Starting VPN Recommender application")
    # You could pre-load models here if they're large
    # Example: load('models/model.pkl')

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    logger.info("Shutting down application")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """Render the main recommendation form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    speed: Optional[float] = Form(None),
    price: Optional[float] = Form(None),
    max_devices: Optional[int] = Form(None),
    logging_policy: Optional[str] = Form(None),
    encryption: Optional[str] = Form(None),
    trial_available: Optional[str] = Form(None),
    country: Optional[str] = Form(None)
):
    try:
        if all(v is None for v in [speed, price, max_devices, logging_policy, encryption, trial_available, country]):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Please provide at least one filter option to get recommendations."
            })

        request_data = VPNRecommendationRequest(
            speed=speed,
            price=price,
            max_devices=max_devices,
            logging_policy=logging_policy,
            encryption=encryption,
            trial_available=trial_available,
            country=country
        )

        inputs = request_data.dict(exclude_none=True)
        if 'trial_available' in inputs:
            inputs['trial_available'] = inputs['trial_available'] == 'yes'

        logger.info(f"Generating recommendations for: {inputs}")
        results = recommend_vpn(inputs)

        # Map raw country names to standardized display names
        country_display_map = {
            "Schweiz/Suisse/Svizzera/Svizra": "Switzerland",
            "USA": "USA",
            "United States": "USA",
            "France": "France",
            "Italia": "Italy",
            "United Kingdom": "UK",
            "Sesel": "Seychelles",
            "Česko": "Czech Republic",
            "Sverige": "Sweden",
            "Danmark": "Denmark",
            "中国": "Hong Kong",
            "Canada": "Canada",
            "Magyarország": "Hungary",
            "Moldova": "Moldova",
            "Australia": "Australia",
            "Deutschland": "Germany",
            "Ísland": "Iceland",
            "România": "Romania",
            "Panamá": "Panama",
            "Κύπρος - Kıbrıs": "Cyprus",
            "British Virgin Islands": "British Virgin Islands",
            "Malaysia مليسيا": "Malaysia",
            "Suomi / Finland": "Finland",
            "Belize": "Belize",
            "Gibraltar": "Gibraltar",
            "Singapore": "Singapore",
            "Bosna i Hercegovina / Босна и Херцеговина": "Bosnia",
            "Norge": "Norway",
            "British Indian Ocean Territory": "British Indian Ocean",
            "Nederland": "Netherlands",
            "Éire / Ireland": "Ireland",
            "臺灣": "Taiwan",
            "Maroc ⵍⵎⵖⵔⵉⴱ المغرب": "Morocco",
            "Mauritius / Maurice": "Mauritius",
            "India": "India",
            "Slovensko": "Slovakia",
            "Barbados": "Barbados",
            "日本": "Japan",
            "Polska": "Poland",
            "Malta": "Malta",
            "България": "Bulgaria",
            "ישראל": "Israel",
            "الإمارات العربية المتحدة": "UAE",
        }

        recommendations = results.to_dict(orient="records")

        # Add a field `country_display` for template use
        for r in recommendations:
            raw = r.get("country", "")
            r["country_display"] = country_display_map.get(raw, raw)



        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": recommendations,
            "inputs": inputs
        })

    except ValueError as e:
        logger.warning(f"Input validation error: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "inputs": {
                "speed": speed,
                "price": price,
                "max_devices": max_devices,
                "logging_policy": logging_policy,
                "encryption": encryption,
                "trial_available": trial_available,
                "country": country
            }
        })
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "We couldn't generate recommendations. Please try again later."
        }, status_code=500)
