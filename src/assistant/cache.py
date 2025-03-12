from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)
client = QdrantClient("localhost:6333")
cache_vectorstore = QdrantVectorStore(
    client=client, collection_name="cache", embedding=embeddings
)
cache_retriever = cache_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})
