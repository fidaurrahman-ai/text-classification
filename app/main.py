from fastapi import FastAPI
from endpoints import sklearn

app = FastAPI()

# Include the routes from sklearn.py
app.include_router(sklearn.router)

@app.get("/")
def read_root():
    return {"message": "Hello World from main.py!"}
