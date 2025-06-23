from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from recommender.recommend import recommend_vpn

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def post_recommendation(
    request: Request,
    speed: float = Form(None),
    price: float = Form(None),
    max_devices: int = Form(None),
    logging_policy: str = Form(None),
    encryption: str = Form(None),
    trial_available: str = Form(None),
    country: str = Form(None)
):
    # Validate all required fields are provided
    if None in [speed, price, max_devices, logging_policy, encryption, trial_available, country]:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please fill in all required fields"
        })
    
    # Additional validation for numeric fields
    if speed <= 0 or price < 0 or max_devices <= 0:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please provide valid values for all fields"
        })

    try:
        inputs = {
            "speed": speed,
            "price": price,
            "trial_available": 1 if trial_available.lower() == "yes" else 0,
            "encryption": encryption,
            "logging_policy": logging_policy,
            "max_devices": max_devices,
            "country": country
        }

        results = recommend_vpn(inputs)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": results.to_dict(orient="records")
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"An error occurred while processing your request: {str(e)}"
        })