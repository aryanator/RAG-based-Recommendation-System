from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag_services.recommend import recommend_products  
from rag_services.rag_recommend import generate_llm_recommendation  

app = FastAPI()

# Allow all origins (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify a list of origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class UserQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3  # Default to top 3 recommendations

@app.post("/recommend")
async def normal_recommendation(user_query: UserQuery):
    """Endpoint to get normal product recommendations"""
    recommendations = recommend_products(user_query.query, user_query.top_k)
    return {"recommendations": recommendations}

@app.post("/rag_recommend")
async def rag_recommendation(user_query: UserQuery):
    """Endpoint to get RAG-based product recommendations with detailed descriptions"""
    recommendations = recommend_products(user_query.query, user_query.top_k)
    rag_output = generate_llm_recommendation(user_query.query, recommendations)
    return {
        "recommendations": recommendations,
        "rag_description": rag_output
    }

# Run the app (use 'uvicorn' from the terminal)
# uvicorn app_name:app --reload
# Example: uvicorn main:app --reload
