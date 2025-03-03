from fastapi import FastAPI
import json
from transformers import pipeline  # Import Hugging Face Transformers
import os

app = FastAPI()

# Load mock data
with open("data/products.json", "r") as f:
    products = json.load(f)

with open("data/sales_data.json", "r") as f:
    sales_data = json.load(f)

# Initialize Hugging Face text generation pipeline with GPT-Neo (125M)
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")  # Use GPT-Neo (125M)

def get_sales_score(product_id):
    """Retrieve sales data for ranking recommendations."""
    sales = next((s for s in sales_data if s["product_id"] == product_id), None)
    if sales:
        total_sales = sum(day["units_sold"] for day in sales["daily_sales"])
        return total_sales  # Higher sales = higher recommendation priority
    return 0

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the AI-Powered Product Recommendation API!"}

@app.get("/recommendations")
def get_recommendations(effect: str):
    """
    Get product recommendations based on user preferences (effect).
    Example: /recommendations?effect=relaxation
    """
    # Filter products based on the effect
    filtered_products = [p for p in products if effect in p["effects"]]

    # Rank products based on sales data (popularity)
    ranked_products = sorted(filtered_products, key=lambda p: get_sales_score(p["id"]), reverse=True)

    return {"effect": effect, "recommendations": ranked_products}

# Load knowledge base
with open("data/knowledge.json", "r") as f:
    knowledge_base = json.load(f)

def get_ingredient_info(ingredients):
    """
    Retrieve additional knowledge about ingredients.
    """
    ingredient_details = []
    for ing in ingredients:
        info = next((k for k in knowledge_base if k["name"].lower() == ing.lower()), None)
        if info:
            ingredient_details.append({
                "name": ing,
                "properties": info["properties"],
                "common_effects": info["common_effects"]
            })
    return ingredient_details

def generate_ai_description(product, enriched_ingredients):
    """
    Generate an engaging product description using a more effective prompt.
    """
    # Simplified prompt
    prompt = f"""
    Write a product description for {product['name']}, a {product['type']} that helps with {', '.join(product['effects'])}.
    Ingredients: {', '.join([f"{ing['name']} ({ing['properties']})" for ing in enriched_ingredients])}.
    Description: {product['description']}
    """

    # Generate text using the selected model
    response = generator(
        prompt,
        max_new_tokens=50,  # Limit the output length
        temperature=0.7,    # Control creativity
        num_return_sequences=1,
        truncation=True     # Ensure the prompt is not too long
    )

    # Clean up the output
    generated_description = response[0]['generated_text'].strip()

    # Remove the prompt from the output (if present)
    if prompt in generated_description:
        generated_description = generated_description.replace(prompt, "").strip()

    # Remove any additional unwanted text
    if "Ingredients:" in generated_description:
        generated_description = generated_description.split("Ingredients:")[0].strip()

    return "Hello"

@app.get("/product-info/{product_id}")
def get_product_info(product_id: int):
    """
    Retrieve product details by ID, augment with ingredient knowledge, and enhance with Hugging Face Transformers.
    """
    product = next((p for p in products if p["id"] == product_id), None)
    if not product:
        return {"error": "Product not found"}
    
    # Retrieve ingredient knowledge
    enriched_ingredients = get_ingredient_info(product["ingredients"])

    # Generate enhanced description with Hugging Face Transformers
    ai_description = generate_ai_description(product, enriched_ingredients)

    return {
        "product": product,
        "enriched_ingredients": enriched_ingredients,
        "ai_generated_description": ai_description
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)