---

# AI Product Recommendation System with RAG

This repository contains an AI-powered product recommendation system with a Retrieval-Augmented Generation (RAG) approach to enhance personalized recommendations. The system integrates FastAPI for the backend, along with a simple frontend to interact with the API.

## Project Structure

The project includes the following components:

- **Backend**: FastAPI server with two main endpoints for normal and RAG-based recommendations.
- **Frontend**: Simple HTML/JavaScript interface to interact with the backend API and fetch recommendations.
- **Recommendation Algorithm**: A basic recommendation algorithm based on product attributes and user queries.
- **RAG Implementation**: Retrieves relevant product information based on user queries and enhances recommendations with the help of an LLM- Leveraged Mistral 7B.

### File Structure

```plaintext
recommendation-system/
├── data/                      # Contains any data files used in the project
├── env/                       # Virtual environment for the project
├── Images/                    # Evidence
├── vector_db/                 # Contains the FAISS index and metadata
├── __pycache__/               # Compiled Python files (auto-generated)
├── create_vector_db.py        # Script to create the FAISS vector database
├── frontend.html              # Frontend interface for user interaction
├── GoogleNews-vectors-negative300.bin  # Pre-trained word vectors (used for embeddings)
├── main.py                    # FastAPI application (contains endpoints for normal and RAG recommendations)
├── rag_recommend.py           # Contains functions for RAG-based recommendation and text generation using LLM
├── recommend.py               # Contains functions for normal product recommendation based on user queries
├── test_llm.py                # Script to test LLM and RAG functionality (optional)

```

### File Descriptions

#### **backend/**

1. **`main.py`**: This file contains the FastAPI server code, which defines the two main endpoints:
   - `POST /recommend`: For normal product recommendations.
   - `POST /rag_recommend`: For RAG-based recommendations (uses LLM to generate detailed descriptions).
   
   The server handles incoming POST requests from the frontend, processes them using the recommendation and RAG algorithms, and returns the recommendations.

2. **`recommend.py`**: This file contains the logic for the normal product recommendation algorithm. It processes the user query, matches it with products based on attributes, and returns the top `k` products as recommendations.

3. **`rag_recommend.py`**: This file contains the logic for the RAG (Retrieval-Augmented Generation) implementation. It uses the normal recommendations from `recommend.py` and passes them through an LLM (e.g., GPT-4) to generate enhanced, detailed descriptions for the products.

#### **frontend/**

1. **`index.html`**: A basic HTML file for the frontend interface. It provides a simple form for users to input their query and select the number of recommendations they want to receive. It sends requests to the backend API when the user clicks the buttons.

### How to Run the Prototype

#### Prerequisites

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

#### Setting Up Hugging Face Token

To interact with the Mistral model, you need a Hugging Face API token. **Do not hardcode your token directly** in the code for security purposes.

- **Option 1: Use a placeholder**  
  Open the `rag_recommend.py` file and replace the `YOUR_HUGGING_FACE_TOKEN` placeholder with your actual Hugging Face token.
  
  ```python
  token = "YOUR_HUGGING_FACE_TOKEN"
  ```

- **Option 2: Use a `.env` file**  
  Alternatively, set up the token in an `.env` file for better security.  
  - Create a file named `.env` in the root directory of the project and add the following line:
  
    ```bash
    HF_TOKEN="your_hugging_face_token"
    ```

  - The application will automatically read this token from the `.env` file when you run it.

#### Running the Backend (FastAPI)

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/recommendation-system.git
   cd recommendation-system
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload
   ```

   This will start the server on `http://127.0.0.1:8000`.

3. You can test the backend API using `POST` requests via a tool like Postman or interact with it using the frontend.

#### Running the Frontend (Simple HTML/JavaScript)

1. Navigate to the `frontend` folder in the project directory and start the frontend server:
   ```bash
   http-server frontend
   ```

   This will serve the frontend on `http://127.0.0.1:8080`.

2. Open the browser and go to `http://127.0.0.1:8080` to interact with the recommendation system.

#### Example Request

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


### Assumptions and Simplifications

1. **Mock Data**: The recommendation system uses mock data for products, which can be expanded to a real dataset.
2. **Simple Recommendation Algorithm**: The current recommendation approach is based on simple product attributes and similarity matching.
3. **No User Authentication**: For simplicity, user authentication is not implemented in this prototype.
4. **Limited Frontend**: The frontend is basic, designed for demonstrating the functionality and not a production-ready solution.
5. **No Database Integration**: The system uses in-memory mock data, and a real-world implementation would likely require a database.

### Approach to Recommendation Algorithm and RAG Implementation

- **Normal Recommendation**: The system matches products based on the given query (e.g., `"something to help me improve my digestion"`) and retrieves the top `k` most relevant products from the mock dataset.
  
- **RAG Implementation**: The Retrieval-Augmented Generation (RAG) approach uses the products retrieved from the normal recommendation step and feeds them to an LLM (e.g., GPT-4) to generate detailed descriptions and improve the recommendations based on user preferences.

### Design Decisions and Trade-offs
- While developing this recommendation system with RAG, several important design decisions and trade-offs were made:

- Model Choice (Mistral 7B Instruct):
We chose the Mistral 7B Instruct model for its capability to generate rich and context-aware product recommendations. However, due to its large size, the model can be slow to run on a CPU, especially with more complex queries. For optimal performance, it's recommended to run the model on a GPU.

- Speed vs. Accuracy:
While faster models may offer reduced latency, they tend to be less accurate, especially in terms of generating meaningful and personalized descriptions. Therefore, we opted for the more accurate, but slower Mistral 7B model, which provides a better balance between recommendation quality and model performance. For real-time applications, a compromise could be considered depending on the specific use case.

- Backend (FastAPI):
FastAPI was chosen due to its simplicity, speed, and flexibility in handling asynchronous tasks. This enables us to handle requests efficiently even though we are working with potentially time-consuming processes like embedding generation and LLM inference.

- Frontend Interaction:
A simple HTML/JavaScript frontend was created to interact with the FastAPI backend. This decision was made for rapid prototyping and ease of demonstration, but for a production-grade application, a more sophisticated frontend (e.g., React or Angular) could be used.

- Mock Data:
For demonstration purposes, we used mock product data and a simple recommendation algorithm. While this serves the prototype's purpose, for a real-world application, connecting to a live product database with real-time updates would be crucial for delivering accurate recommendations.

- Model and System Scalability:
While this system works for smaller datasets and local use cases, scaling it for large-scale production would require optimizations in both data storage (e.g., using cloud solutions) and model performance (e.g., distributed inference for LLMs).

- These trade-offs were necessary to balance the quality of the recommendations, the computational resources available, and the speed at which the system needs to operate.
  
### Areas for Improvement or Expansion

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
