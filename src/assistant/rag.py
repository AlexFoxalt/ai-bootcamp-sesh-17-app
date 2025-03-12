from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)
client = QdrantClient("localhost:6333")
vectorstore = QdrantVectorStore(
    client=client, collection_name="DnD_Documents", embedding=embeddings
)
rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
