import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load product data from JSON file
with open("data/products.json", "r") as file:
    products = json.load(file)

# Load a lightweight embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create text representations for embeddings
product_texts = [
    f"{p['name']}. {p['description']} Effects: {', '.join(p['effects'])}. Ingredients: {', '.join(p['ingredients'])}."
    for p in products
]

# Generate embeddings
product_embeddings = embedding_model.encode(product_texts)

# Create FAISS index
index = faiss.IndexFlatL2(product_embeddings.shape[1])
index.add(np.array(product_embeddings))

# Save FAISS index
faiss.write_index(index, "vector_db/faiss_index.bin")

# Store product metadata separately
with open("vector_db/product_metadata.pkl", "wb") as f:
    pickle.dump(products, f)

print("✅ FAISS index and metadata saved!")
