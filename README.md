# VPN Recommender AI

## Setup

```bash
pip install -r requirements.txt
python models/train_model.py
uvicorn app.main:app --reload
