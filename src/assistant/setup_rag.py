from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_qdrant import QdrantVectorStore


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
    )
    json_loader = DirectoryLoader(
        path="./data/data",
        glob="**/*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": "..", "text_content": False},
    )
    json_documents = json_loader.load()
    qdrant = QdrantVectorStore.from_documents(
        json_documents,
        embeddings,
        url="localhost:6333",
        collection_name="DnD_Documents",
    )
