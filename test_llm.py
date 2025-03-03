from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# Load T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # Set legacy=False

# Sample product data (replace with your actual JSON data)
product_data = [
    {
        "id": 1,
        "name": "Relaxation Tea",
        "type": "beverage",
        "description": "A soothing herbal tea blend.",
        "effects": ["relaxation", "stress relief"],
        "ingredients": ["Chamomile", "Lavender"],
        "price": 12.99
    }
]

# Sample ingredient data (replace with your actual JSON data)
ingredient_data = [
    {
        "name": "Chamomile",
        "properties": "Mild, floral aroma; known for calming effects",
        "common_effects": ["relaxation", "improved sleep"]
    },
    {
        "name": "Lavender",
        "properties": "Soothing fragrance; commonly used for relaxation and stress relief",
        "common_effects": ["stress relief", "relaxation"]
    }
]

# Extract product and ingredient details
product = product_data[0]
ingredients = ingredient_data

# Build the prompt dynamically
ingredient_details = "\n".join([
    f"- {ingredient['name']}: {ingredient['properties']}"
    for ingredient in ingredients
])

# Simplified prompt
prompt = f"""
Write a short, engaging product description for {product['name']}, a {product['type']}
made with {', '.join(product['ingredients'])}. It helps with {', '.join(product['effects'])}.
The ingredients include:
{ingredient_details}
"""

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text using T5
outputs = model.generate(
    input_ids,
    max_length=100,  # Limit output length
    num_return_sequences=1,  # Generate only one sequence
    no_repeat_ngram_size=2  # Prevent repetition of phrases
)

# Decode the generated text
generated_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result
print("Generated Description:")
print(generated_description)