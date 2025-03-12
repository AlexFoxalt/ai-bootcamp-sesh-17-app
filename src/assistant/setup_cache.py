from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
    )
    client = QdrantClient("localhost:6333")
    if client.collection_exists("cache"):
        client.delete_collection("cache")
    client.create_collection(
        collection_name="cache",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
