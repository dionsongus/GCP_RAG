# This app can load docs and respond to queries based on the doc using Vertex AI.

import os
import json
from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

app = Flask(__name__)

# --- Environment Variables ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cogent-theater-294521")
REGION = os.getenv("GCP_REGION", "us-central1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "cogent-theater-294521-rag-app-gcs-bucket")

# In‑memory document store
document_store = []


# ---------------------------------------------------------
# 1. LAZY INITIALIZATION HELPERS
# lazy initialization is about pushing the service/dependency init. to the routes to help debugging.
# ---------------------------------------------------------

def get_storage_client():
    """Create a Storage client only when needed."""
    return storage.Client(project=PROJECT_ID)


def get_bucket():
    client = get_storage_client()
    return client.bucket(GCS_BUCKET_NAME)


def get_embedding_client():
    """Create embedding client lazily."""
    return aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )


def get_embedding_model_path():
    """Resolve model path lazily."""
    return aiplatform.get_model_garden_model_path(
        "text-embedding-004",
        project=PROJECT_ID,
        location=REGION
    )


def get_llm():
    """Create LLM client lazily."""
    return aiplatform.GenerativeModel("gemini-1-5-flash-001")


# ---------------------------------------------------------
# 2. EMBEDDING + DOCUMENT LOADING
# ---------------------------------------------------------

def get_embedding(text):
    if not text:
        return []

    try:
        embedding_client = get_embedding_client()
        model_path = get_embedding_model_path()

        response = embedding_client.predict(
            endpoint=model_path,
            instances=[{"content": text}],
        )
        return response.predictions[0]["embeddings"]["values"]

    except Exception as e:
        app.logger.error(f"Error getting embedding: {e}")
        return []


def load_and_embed_documents():
    """Load documents from GCS and embed them."""
    try:
        bucket = get_bucket()
        blobs = bucket.list_blobs()

        for blob in blobs:
            if blob.name.endswith(".txt"):
                content = blob.download_as_text()
                embedding = get_embedding(content)

                if embedding:
                    document_store.append({
                        "text": content,
                        "embedding": embedding
                    })
                    app.logger.info(f"Embedded document: {blob.name}")

        app.logger.info(f"Finished loading {len(document_store)} documents.")

    except Exception as e:
        app.logger.error(f"Error loading documents: {e}")


def find_similar_documents(query_embedding, top_k=3):
    if not document_store or not query_embedding:
        return []

    similarities = []
    for doc in document_store:
        sim = sum(q * d for q, d in zip(query_embedding, doc["embedding"]))
        similarities.append((sim, doc["text"]))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [text for sim, text in similarities[:top_k]]


# ---------------------------------------------------------
# 3. ROUTES
# ---------------------------------------------------------

@app.route("/")
def index():
    return "RAG application is running!"


@app.route("/embed_documents", methods=["POST"])
def embed_docs_endpoint():
    load_and_embed_documents()
    return jsonify({
        "status": "Documents loaded and embedded",
        "count": len(document_store)
    })


@app.route("/query", methods=["POST"])
def query_rag():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # 1. Embed query
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return jsonify({"error": "Failed to generate query embedding"}), 500

    # 2. Retrieve documents
    relevant_docs = find_similar_documents(query_embedding, top_k=2)

    if not relevant_docs:
        prompt_text = f"Answer the following question: {user_query}"
        app.logger.warning("No relevant documents found.")
    else:
        context = "\n\n".join(relevant_docs)
        prompt_text = (
            "You are a helpful assistant. Use the following context to answer the question:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            "Answer:"
        )

    # 3. Generate LLM response
    try:
        llm = get_llm()
        response = llm.generate_content(
            prompt_text,
            generation_config={"temperature": 0.2, "max_output_tokens": 500}
        )
        answer = response.text

    except Exception as e:
        app.logger.error(f"Error generating LLM response: {e}")
        answer = "I encountered an error generating a response."

    return jsonify({
        "query": user_query,
        "answer": answer,
        "retrieved_context": relevant_docs
    })


# ---------------------------------------------------------
# 4. LOCAL DEV ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

