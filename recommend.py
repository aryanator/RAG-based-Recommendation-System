import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("vector_db/faiss_index.bin")

# Load product metadata
with open("vector_db/product_metadata.pkl", "rb") as f:
    products = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_products(user_query, top_k=3):
    """Retrieve top-k most similar products using FAISS"""
    # Convert user query into an embedding
    query_embedding = embedding_model.encode([user_query])

    # Search for top-k most similar products
    _, top_indices = index.search(np.array(query_embedding), top_k)

    # Fetch recommended products
    recommendations = [products[idx] for idx in top_indices[0]]
    
    return recommendations

# Example test case
if __name__ == "__main__":
    user_query = "I need something to help me sleep after a long workday."
    recommended_products = recommend_products(user_query)

    # Print recommendations
    print("\nRecommended Products:")
    for product in recommended_products:
        print(f"- {product['name']}: {product['description']}")
