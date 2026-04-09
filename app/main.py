from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from app.services import run_prediction

@asynccontextmanager
async def lifespan(app):
    logging.info("🚀 API starting up...")
    yield
    logging.info("🛑 API shutting down...")

app = FastAPI(title='Churn Prediction API',lifespan=lifespan)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str


@app.get('/')
def home():
    return {'message':'Churn Prediction API is running'}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post('/predict')
def predict_churn(data: CustomerData):

    try:
        logging.info(f"Received request: {data}")

        # 🔹 Validation
        if data.tenure < 0:
            logging.warning("Invalid input: Negative tenure")
            raise HTTPException(status_code=400, detail="Tenure cannot be negative")

        if data.MonthlyCharges < 0:
            logging.warning("Invalid input: Negative MonthlyCharges")
            raise HTTPException(status_code=400, detail="Monthly Charges cannot be negative")

        if data.TotalCharges < 0:
            logging.warning("Invalid input: Negative TotalCharges")
            raise HTTPException(status_code=400, detail="Total Charges cannot be negative")

        # 🔹 Input preparation
        input_data = {
            'gender': data.gender,
            'SeniorCitizen': data.SeniorCitizen,
            'Partner': data.Partner,
            'Dependents': data.Dependents,
            'tenure': data.tenure,
            'MonthlyCharges': data.MonthlyCharges,
            'TotalCharges': data.TotalCharges,
            'Contract': data.Contract,
            'InternetService': data.InternetService,
            'PaymentMethod': data.PaymentMethod
        }

        logging.info(f"Processed input: {input_data}")

        # 🔹 Prediction
        result = run_prediction(input_data)

        logging.info(f"Prediction result: {result}")

        return result

    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")