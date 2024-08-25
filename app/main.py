from fastapi import FastAPI
from app.endpoints.sklearn import router as logistic_regression_router

app = FastAPI()

# Include the logistic regression router
app.include_router(logistic_regression_router, prefix="/sklearn/logistic-regression", tags=["Logistic Regression"])

# You can add other routers or endpoints here

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
