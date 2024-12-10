from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Optional, List, Dict, Any

app = FastAPI()

#loggin init
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#loading the models
models = {
    'RandomForest': joblib.load('models/RandomForest_model.pkl'),
    'SVM': joblib.load('models/SVM_model.pkl'),
    'LogisticRegression': joblib.load('models/LogisticRegression_model.pkl'),
    'NaiveBayes': joblib.load('models/NaiveBayes_model.pkl'),
    'XGBoost': joblib.load('models/XGBoost_model.pkl')
}

class APIEndpoint(BaseModel):
    path: str
    method: str
    description: str
    request_body: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None

#loading the features and encodings in order
feature_names = ['amount', 'distance_from_home', 'auth_attempts', 'user_archetype_encoded', 'merchant_category_encoded', 'transaction_type_encoded']
logger.info(f"Loaded feature names: {feature_names}")
with open('models/encodings.pkl', 'rb') as f:
    encodings = pickle.load(f)

class Transaction(BaseModel): #pydantic model for request
    transaction_id: str
    amount: float
    timestamp: str
    merchant_name: str
    merchant_category: str
    transaction_type: str
    distance_from_home: float
    device_type: str
    is_trusted_device: bool
    three_d_secure: str
    auth_attempts: int
    user_archetype: str
    is_fraud: bool

class PredictionResponse(BaseModel): #pydantic model for response
    model: str
    prediction: str
    probabilities: Optional[List[float]] = None

def inspect_model_features(): #function to inspect the features of the models to fix a specific error on the user_archetype_encoded
    for model_name, model in models.items():
        try:
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model {model_name} expects features: {model.feature_names_in_}")
            else:
                logger.info(f"Model {model_name} does not have attribute 'feature_names_in_'")
        except Exception as e:
            logger.error(f"Error inspecting model {model_name}: {e}")

def preprocess(transaction: Transaction):
    try:
        #encode the categorical columns
        user_archetype_encoded = encodings['user_archetype'].get_loc(transaction.user_archetype)
        merchant_category_encoded = encodings['merchant_category'].get_loc(transaction.merchant_category)
        transaction_type_encoded = encodings['transaction_type'].get_loc(transaction.transaction_type)
        
        #creating the features
        features = pd.DataFrame({
            'amount': [transaction.amount],
            'distance_from_home': [transaction.distance_from_home],
            'auth_attempts': [transaction.auth_attempts],
            'user_archetype_encoded': [user_archetype_encoded],
            'merchant_category_encoded': [merchant_category_encoded],
            'transaction_type_encoded': [transaction_type_encoded],
        })
        
        #checking if the feature names are in order
        features = features.reindex(columns=feature_names, fill_value=0)
        logger.info(f"Preprocessed features: {features}")
        return features
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def make_prediction(model_name: str, features: pd.DataFrame):
    try:
        model = models[model_name]
        logger.info(f"Features passed to model {model_name}: {features.columns.tolist()}")
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]
            logger.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
            return {
                "prediction": "Fraud" if prediction == 1 else "Not Fraud",
                "probabilities": probabilities.tolist()
            }
        else:
            prediction = model.predict(features)[0]
            logger.info(f"Prediction: {prediction}")
            return {
                "prediction": "Fraud" if prediction == 1 else "Not Fraud"
            }
    except Exception as e:
        logger.error(f"Error in making prediction: {e}")
        raise

#api endpoints
@app.get("/")
async def get_api_documentation():
    """Returns documentation about available API endpoints"""
    example_transaction = {
        "transaction_id": "TX123",
        "amount": 499.99,
        "timestamp": "2024-03-20 14:30:00",
        "merchant_name": "Example Store",
        "merchant_category": "Retail",
        "transaction_type": "online",
        "distance_from_home": 5.2,
        "device_type": "mobile",
        "is_trusted_device": True,
        "three_d_secure": "authenticated",
        "auth_attempts": 1
    }

    example_response = {
        "model": "RandomForest",
        "prediction": "Not Fraud",
        "probabilities": [0.98, 0.02]
    }

    return {
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Returns API documentation with available endpoints"
            },
            {
                "path": "/api/randomforest",
                "method": "POST",
                "description": "Makes fraud prediction using Random Forest model",
                "request_body": example_transaction,
                "response": example_response
            },
            {
                "path": "/api/svm",
                "method": "POST", 
                "description": "Makes fraud prediction using SVM model",
                "request_body": example_transaction,
                "response": example_response
            },
            {
                "path": "/api/logisticregression",
                "method": "POST",
                "description": "Makes fraud prediction using Logistic Regression model",
                "request_body": example_transaction,
                "response": example_response
            },
            {
                "path": "/api/naivebayes",
                "method": "POST",
                "description": "Makes fraud prediction using Naive Bayes model",
                "request_body": example_transaction,
                "response": example_response
            },
            {
                "path": "/api/xgboost",
                "method": "POST",
                "description": "Makes fraud prediction using XGBoost model",
                "request_body": example_transaction,
                "response": example_response
            }
        ],
        "available_models": list(models.keys()),
        "required_features": feature_names
    }

@app.post("/api/randomforest", response_model=PredictionResponse)
async def predict_randomforest(transaction: Transaction):
    try:
        features = preprocess(transaction)
        result = make_prediction('RandomForest', features)
        return PredictionResponse(model="Random Forest", prediction=result["prediction"], probabilities=result.get("probabilities"))
    except Exception as e:
        logger.error(f"Error in /api/randomforest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/svm", response_model=PredictionResponse)
async def predict_svm(transaction: Transaction):
    try:
        features = preprocess(transaction)
        result = make_prediction('SVM', features)
        return PredictionResponse(model="SVM", prediction=result["prediction"], probabilities=result.get("probabilities"))
    except Exception as e:
        logger.error(f"Error in /api/svm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logisticregression", response_model=PredictionResponse)
async def predict_logistic(transaction: Transaction):
    try:
        features = preprocess(transaction)
        result = make_prediction('LogisticRegression', features)
        return PredictionResponse(model="Logistic Regression", prediction=result["prediction"], probabilities=result.get("probabilities"))
    except Exception as e:
        logger.error(f"Error in /api/logisticregression: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/naivebayes", response_model=PredictionResponse)
async def predict_naivebayes(transaction: Transaction):
    try:
        features = preprocess(transaction)
        result = make_prediction('NaiveBayes', features)
        return PredictionResponse(model="Naive Bayes", prediction=result["prediction"], probabilities=result.get("probabilities"))
    except Exception as e:
        logger.error(f"Error in /api/naivebayes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/xgboost", response_model=PredictionResponse)
async def predict_xgboost(transaction: Transaction):
    try:
        features = preprocess(transaction)
        result = make_prediction('XGBoost', features)
        return PredictionResponse(model="XGBoost", prediction=result["prediction"], probabilities=result.get("probabilities"))
    except Exception as e:
        logger.error(f"Error in /api/xgboost: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
inspect_model_features()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)