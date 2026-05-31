from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import motor.motor_asyncio
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Loan Approval Predictor API")

# Setup CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client.loan_approval_db
predictions_collection = db.predictions

# Load Model and Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    model = None
    scaler = None

class LoanApplication(BaseModel):
    no_of_dependents: int
    education: int # 0 for Not Graduate, 1 for Graduate
    self_employed: int # 0 for No, 1 for Yes
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

@app.get("/")
def read_root():
    return {"message": "Welcome to Loan Approval Predictor API"}

@app.post("/predict")
async def predict_loan(application: LoanApplication):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not loaded. Please train the model first.")
        
    try:
        # Convert input to DataFrame to match training data structure
        input_data = pd.DataFrame([application.model_dump()])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction_num = model.predict(input_scaled)[0]
        prediction_prob = model.predict_proba(input_scaled)[0][prediction_num]
        
        # In the Kaggle dataset, usually 0/1 are encoded. 
        # Typically, 0 is Approved and 1 is Rejected, but this depends on LabelEncoder.
        # Let's assume standard output mapping. You may need to verify with EDA.
        status = "Approved" if prediction_num == 0 else "Rejected"
        
        # Save to database
        record = application.model_dump()
        record['prediction'] = status
        record['probability'] = float(prediction_prob)
        record['timestamp'] = datetime.now()
        
        await predictions_collection.insert_one(record)
        
        return {
            "prediction": status,
            "probability": float(prediction_prob)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
