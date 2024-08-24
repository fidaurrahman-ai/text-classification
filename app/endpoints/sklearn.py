from fastapi import APIRouter

router = APIRouter()

@router.get("/sklearn")
def read_sklearn():
    return {"message": "Hello World from sklearn.py!"}
