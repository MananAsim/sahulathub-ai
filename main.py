# -*- coding: utf-8 -*-
"""
main.py — SahulatHub AI Microservice

FastAPI service that exposes the recommendation engine over HTTP.
Model and embeddings load ONCE at startup (when recommendation_engine is imported).

Start:
    uvicorn main:app --reload --port 8001

Node backend calls:
    POST http://localhost:8001/match
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ─── Import engine ────────────────────────────────────────────────────────────
# Importing this module triggers model loading and embedding precomputation.
# This happens exactly ONCE when the server process starts.
from recommendation_engine import recommend


# ─── Request / Response schemas ───────────────────────────────────────────────

class MatchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Job description, e.g. 'fix leaking pipe'")
    lat: float = Field(..., ge=-90, le=90, description="Client latitude")
    lng: float = Field(..., ge=-180, le=180, description="Client longitude")
    radius: float = Field(default=50.0, gt=0, description="Search radius in km")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class MatchedWorker(BaseModel):
    worker_id: str
    primary_skill: str
    final_score: float
    distance_km: float
    rating: float


class MatchResponse(BaseModel):
    success: bool
    count: int
    data: list[MatchedWorker]


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SahulatHub AI Matchmaking",
    description="Context-aware hybrid matchmaking microservice for SahulatHub FYP",
    version="1.0.0",
)

# Allow CORS from Node backend and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "success": True,
        "service": "SahulatHub AI Matchmaking",
        "status": "running",
    }


# ─── Match endpoint ──────────────────────────────────────────────────────────

@app.post("/match", response_model=MatchResponse)
def match_workers(req: MatchRequest):
    """
    Receives a job query + location + radius from the Node backend.
    Returns top matched workers sorted by final_score.
    """
    try:
        results = recommend(
            user_query=req.query,
            user_lat=req.lat,
            user_lng=req.lng,
            radius_km=req.radius,
            top_n=req.top_n,
        )

        # Ensure worker_id is always a string for JSON consistency
        for r in results:
            r["worker_id"] = str(r["worker_id"])

        return MatchResponse(
            success=True,
            count=len(results),
            data=results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Direct execution ────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
