import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Patch Priority Predictor API")


# Templates folder
templates = Jinja2Templates(directory="templates")



# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("front.html", {"request": request})


class SystemData(BaseModel):
    Regulatory_Zone: str
    Criticality_Level: int
    Uptime_days: int
    Last_Patch_Date: str
    Next_Available_Window: str
    Downtime_Window_hours: int
    Requires_Reboot: str
    Vulnerability_Score: float
    Predicted_Failure_Probability: float

def preprocess_input(data: SystemData):
    today = datetime.now()
    last_patch_age = (today - datetime.strptime(data.Last_Patch_Date, "%Y-%m-%d")).days
    days_until_next = (datetime.strptime(data.Next_Available_Window, "%Y-%m-%d") - today).days

    zone_b = 1 if data.Regulatory_Zone == "Zone B" else 0
    zone_c = 1 if data.Regulatory_Zone == "Zone C" else 0
    reboot_yes = 1 if data.Requires_Reboot.lower() == "yes" else 0

    df = pd.DataFrame([{
        "Criticality Level (1-5)": data.Criticality_Level,
        "Uptime (days)": data.Uptime_days,
        "Downtime Window (hours)": data.Downtime_Window_hours,
        "Vulnerability Score (0-10)": data.Vulnerability_Score,
        "Predicted Failure Probability": data.Predicted_Failure_Probability,
        "Last Patch Age (days)": last_patch_age,
        "Days Until Next Window": days_until_next,
        "Regulatory Zone_Zone B": zone_b,
        "Regulatory Zone_Zone C": zone_c,
        "Requires Reboot_Yes": reboot_yes
    }])

    # Reorder columns to match training
    df = df[model_columns]
    return df



@app.post("/predict")
def predict(data: SystemData):
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    return {"Predicted Patch Priority": prediction}

