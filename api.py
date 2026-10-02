from fastapi import FastAPI
from pydantic import BaseModel
import model_pipeline

app = FastAPI(
    title="NNDL Movie Sentiment Analyser API",
    version="1.0"
)


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def home():
    return {
        "message": "Movie Sentiment Analyser API is running"
    }


@app.post("/predict")
def predict(request: ReviewRequest):

    review = request.review.strip()

    if not review:
        return {
            "error": "Please enter a movie review."
        }

    results = model_pipeline.predict_all_models(review)

    return {
        "review": review,
        "results": results
    }