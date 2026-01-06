# This app can load docs and respond to queries based on the doc using Vertex AI.


import os
import json
from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import aiplatform
from dotenv import load_dotenv # For local development

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)





# 1.0 --- Setups ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cogent-theater-294521") # get project id from the env var first, use the hard coded value as a fall back
REGION = os.getenv("GCP_REGION", "us-central1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "cogent-theater-294521-rag-app-gcs-bucket")

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

# Initialize Storage Client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# --- Vertex AI Models ---
# Text Embedding Model
EMBEDDING_MODEL_NAME = "text-embedding-004" # Or "text-embedding-gecko" for older versions
embedding_model = aiplatform.get_model_garden_model_path(
    EMBEDDING_MODEL_NAME, project=PROJECT_ID, location=REGION
)
embedding_client = aiplatform.gapic.PredictionServiceClient(
    client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
)

# Generative Language Model (LLM)
# Using Gemini 1.5 Flash - adjust as needed (e.g., "gemini-pro")
LLM_MODEL_NAME = "gemini-1-5-flash-001"
llm = aiplatform.GenerativeModel(LLM_MODEL_NAME)





# 2.0 --- Define RAG Components ---

# Simplified In-Memory Document Store (for demo purposes)
# In a real application, you'd use a dedicated vector database (e.g., AlloyDB for PostgreSQL with pgvector, Redis, Pinecone, Weaviate, etc.)
document_store = [] # Stores {'text': '...', 'embedding': [...] }

# Function to get embeddings from Vertex AI
def get_embedding(text):
    if not text:
        return []
    try:
        response = embedding_client.predict(
            endpoint=embedding_model,
            instances=[{"content": text}],
        )
        return response.predictions[0]["embeddings"]["values"]
    except Exception as e:
        app.logger.error(f"Error getting embedding: {e}")
        return []

# Function to load and embed documents (this would be an offline process)
def load_and_embed_documents():
    app.logger.info(f"Loading documents from gs://{GCS_BUCKET_NAME}...")
    blobs = storage_client.list_blobs(GCS_BUCKET_NAME)
    for blob in blobs:
        if blob.name.endswith(".txt"):
            content = blob.download_as_text()
            # For simplicity, we'll embed the whole document.
            # In real RAG, you'd chunk the text first.
            embedding = get_embedding(content)
            if embedding:
                document_store.append({'text': content, 'embedding': embedding})
                app.logger.info(f"Embedded document: {blob.name}")
    app.logger.info(f"Finished loading {len(document_store)} documents.")

# Similarity Search (very basic dot product for demo)
def find_similar_documents(query_embedding, top_k=3):
    if not document_store or not query_embedding:
        return []

    similarities = []
    for doc in document_store:
        # Calculate dot product (simple similarity)
        similarity = sum(q * d for q, d in zip(query_embedding, doc['embedding']))
        similarities.append((similarity, doc['text']))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [text for sim, text in similarities[:top_k]]




# 3.0 --- Define Flask Routes ---

@app.route('/')
def index():
    return "RAG application is running!"

@app.route('/embed_documents', methods=['POST'])
def embed_docs_endpoint():
    """
    Triggers the document loading and embedding process.
    In a real app, this would be an admin endpoint or a scheduled job.
    """
    load_and_embed_documents()
    return jsonify({"status": "Documents loaded and embedded", "count": len(document_store)})


@app.route('/query', methods=['POST'])
def query_rag():
    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # 1. Embed the user query
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return jsonify({"error": "Failed to generate query embedding"}), 500

    # 2. Retrieve relevant documents
    relevant_docs = find_similar_documents(query_embedding, top_k=2)

    if not relevant_docs:
        # Fallback if no relevant documents are found
        prompt_text = f"Answer the following question: {user_query}"
        app.logger.warning("No relevant documents found for the query.")
    else:
        # 3. Construct prompt with context
        context = "\n\n".join(relevant_docs)
        prompt_text = (
            f"You are a helpful assistant. Use the following context to answer the question:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            f"Answer:"
        )

    # 4. Generate response using LLM
    try:
        response = llm.generate_content(
            prompt_text,
            generation_config={"temperature": 0.2, "max_output_tokens": 500}
        )
        answer = response.text
    except Exception as e:
        app.logger.error(f"Error generating LLM response: {e}")
        answer = "I apologize, but I encountered an error when trying to generate a response."

    return jsonify({
        "query": user_query,
        "answer": answer,
        "retrieved_context": relevant_docs # Optional: show what context was used
    })

if __name__ == '__main__':
    # For local development, set environment variables
    # or ensure you have a .env file with GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET_NAME
    # and authenticate with `gcloud auth application-default login`
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


