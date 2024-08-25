import joblib
from fastapi import APIRouter
from pydantic import BaseModel
from utils.pre_processing import preprocess_text

import os

# Define the input data model
class Review(BaseModel):
    text: str

# Define the path to the model file
model_path = os.path.join(os.path.dirname(__file__), "../../utils/models/Sklearn/Logistic-regression.joblib")

# Load the saved model
model = joblib.load(model_path)

# Define the FastAPI router
router = APIRouter()

# Define the prediction endpoint
@router.post("/predict/")
async def predict_sentiment(review: Review):
    # Preprocess the input text (if necessary)
    text = review.text 
    processed_review = preprocess_text(text) 
    
    # Make a prediction using the loaded model
    prediction = model.predict([processed_review])[0]
    prediction = int(prediction)
    
    # Return the prediction as a response
    return {"sentiment": prediction}
