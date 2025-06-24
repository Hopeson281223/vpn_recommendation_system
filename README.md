## VPN Recommendendation System with AI

An AI-powered system to recommend VPN services based on speed, price, encryption, logging policies, and more. The project includes data cleaning, intelligent enrichment (via NLP and geolocation), and a FastAPI-powered web app for recommendations.

## Features
 - Cleans and standardizes messy VPN datasets
 - Resolves ambiguous country names using geolocation
 - Uses NLP to infer VPN logging policies (DistilBERT-based classifier)
 - Recommends VPNs based on user preferences
 - Interactive web interface built with FastAPI
 - Model training included for VPN ranking

## Setup Instructions
1. Clone the Repository

   """bash"""
    
    git clone https://github.com/Hopeson281223/vpn_recommendation_system.git
    cd vpn_recommendation_system

2. Install Dependencies

   """bash"""

   pip install -r requirements.txt"

3. Train the Recommendation Model

    """bash"""

    python models/train_model.py

4. Run the Web App

    """bash"""

    uvicorn app.main:app --reload
    Visit: http://localhost:8000

## 📁 Project Structure
    vpn_recommendation_system/
    │
    ├── app/                   # FastAPI web app
    │   ├── main.py            # Main entry point for the API
    │   ├── templates/         # Jinja2 templates for HTML pages
    │   └── static/            # CSS or JavaScript files
    │
    ├── data/                       #Raw and Cleaned VPN dataset
    │   ├── data_preprocessing.py         
    │  
    ├── models/                # Machine learning model training
    │   └── train_model.py     # VPN ranking model training script
    │
    ├── recommender/           # Core recommendation logic
    │   └── recommend.py
    │
    ├── utils/                 # Helper functions for cleaning and processing
    │
    ├── requirements.txt       # Python dependencies
    
## Example Usage (API)
Send a POST request to /recommend with parameters:
    json

    {
    "speed": 100,
    "price": 5,
    "trial_available": "yes",
    "encryption": "AES-256",
    "logging_policy": "no_logs",
    "max_devices": 5,
    "country": USA
    }
You’ll receive a list of recommended VPN services matching your preferences.

## AI Components
## Component	        Description
    geopy.Nominatim	    Standardizes and resolves countries
    transformers NLP	Classifies logging policies via text input
    scikit-learn    	Model training for recommendation scoring

📝 License
MIT License. Feel free to fork, use, or contribute to this project.