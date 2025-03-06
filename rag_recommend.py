import json
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
huggingface_token = "YOUR_HUGGING_FACE_TOKEN"  # Replace with your actual token

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=huggingface_token, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Load Knowledge Base
with open("data/knowledge.json", "r") as file:
    ingredient_knowledge = json.load(file)

# Load Product Metadata (FAISS already used for retrieval)
with open("vector_db/product_metadata.pkl", "rb") as f:
    products = pickle.load(f)

def get_ingredient_info(ingredients):
    """Retrieve knowledge about ingredients from knowledge base"""
    return [
        {
            "name": ing,
            "properties": ingredient_knowledge.get(ing, {}).get("properties", "Unknown"),
            "common_effects": ingredient_knowledge.get(ing, {}).get("common_effects", [])
        }
        for ing in ingredients if ing in ingredient_knowledge
    ]

def generate_llm_recommendation(user_query, retrieved_products):
    """Generate a refined recommendation using a local LLM"""

    # Enrich products with ingredient knowledge
    enriched_products = [
        {
            **product,
            "enriched_ingredients": get_ingredient_info(product["ingredients"])
        }
        for product in retrieved_products
    ]

    # Format the prompt
    prompt = f"""
    You are an AI assistant providing personalized product recommendations.

    User Query: "{user_query}"

    Recommended Products:
    {json.dumps(enriched_products, indent=2)}

    Based on the user’s query and the knowledge of ingredients, generate a friendly and persuasive recommendation.
    """

    # Generate response with max_new_tokens
    response = generator(prompt, max_new_tokens=100, temperature=0.7, num_return_sequences=1)
    
    return response[0]['generated_text'].strip()

# Example test case
if __name__ == "__main__":
    user_query = "I want a tea that helps me sleep after a long workday."

    # Import FAISS-based recommendation function
    from recommend import recommend_products
    retrieved_products = recommend_products(user_query)

    # Generate LLM-enhanced output
    llm_output = generate_llm_recommendation(user_query, retrieved_products)

    # Print results
    print("\nGenerated Recommendation:\n", llm_output)
