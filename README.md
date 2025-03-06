---

# AI Product Recommendation System with RAG

This repository contains an AI-powered product recommendation system with a Retrieval-Augmented Generation (RAG) approach to enhance personalized recommendations. The system integrates FastAPI for the backend, along with a simple frontend to interact with the API.

## Project Structure

The project includes the following components:

- **Backend**: FastAPI server with two main endpoints for normal and RAG-based recommendations.
- **Frontend**: Simple HTML/JavaScript interface to interact with the backend API and fetch recommendations.
- **Recommendation Algorithm**: A basic recommendation algorithm based on product attributes and user queries.
- **RAG Implementation**: Retrieves relevant product information based on user queries and enhances recommendations with the help of an LLM.

## How to Run the Prototype

### Prerequisites

1. Python 3.x
2. Node.js and npm (for frontend setup)
3. Install required Python dependencies:
   ```bash
   pip install fastapi uvicorn transformers
   ```

4. Install `http-server` globally to serve the frontend:
   ```bash
   npm install -g http-server
   ```

### Running the Backend (FastAPI)

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/recommendation-system.git
   cd recommendation-system
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

   This will start the server on `http://127.0.0.1:8000`.

3. You can test the backend API using `POST` requests via a tool like Postman, or interact with it using the frontend.

### Running the Frontend (Simple HTML/JavaScript)

1. Navigate to the `frontend` folder in the project directory and start the frontend server:
   ```bash
   http-server
   ```

   This will serve the frontend on `http://127.0.0.1:8080`.

2. Open the browser and go to `http://127.0.0.1:8080` to interact with the recommendation system.

### Example Request

- **Normal Recommendation**:
   Send a `POST` request to `http://127.0.0.1:8000/recommend` with the following JSON body:
   ```json
   {
     "query": "I need something to help me improve my digestion.",
     "top_k": 3
   }
   ```

- **RAG Recommendation**:
   Send a `POST` request to `http://127.0.0.1:8000/rag_recommend` with the same format to get the enhanced recommendations with detailed descriptions.

## Assumptions and Simplifications

1. **Mock Data**: The recommendation system uses mock data for products, which can be expanded to a real dataset.
2. **Simple Recommendation Algorithm**: The current recommendation approach is based on simple product attributes and similarity matching.
3. **No User Authentication**: For simplicity, user authentication is not implemented in this prototype.
4. **Limited Frontend**: The frontend is basic, designed for demonstrating the functionality and not a production-ready solution.
5. **No Database Integration**: The system uses in-memory mock data, and a real-world implementation would likely require a database.

## Approach to Recommendation Algorithm and RAG Implementation

- **Normal Recommendation**: The system matches products based on the given query (e.g., `"something to help me improve my digestion"`) and retrieves the top `k` most relevant products from the mock dataset.
  
- **RAG Implementation**: The Retrieval-Augmented Generation (RAG) approach uses the products retrieved from the normal recommendation step and feeds them to an LLM (e.g., GPT-4) to generate detailed descriptions and improve the recommendations based on user preferences.

## Areas for Improvement or Expansion

1. **Enhanced Recommendation Algorithm**: 
   - Implement more advanced recommendation algorithms, such as collaborative filtering or matrix factorization.
   - Use user behavior data (e.g., clicks, purchases) to improve recommendations.
   
2. **Real Product Data**: Replace mock data with a real product dataset or integrate with a product database.
   
3. **Integrating User Authentication**: Implement user authentication for personalized recommendations.
   
4. **Optimize RAG Generation**: Improve the LLM-based description generation for more coherent and personalized outputs.
   
5. **Scalability**: Implement a distributed architecture for handling large-scale product data and user queries.
   
6. **User Interface Improvements**: Enhance the frontend with modern UI/UX frameworks (e.g., React, Vue.js) and add more interactive features.

## License

This project is licensed under the MIT License.

---
