import chromadb
from chromadb.config import Settings


class ChromaDBHandler:
    def __init__(self, persist_directory="chroma_db"):
        """
        Initializes the ChromaDB client with persistence settings.
        
        Args:
            persist_directory (str): Directory to store ChromaDB collections.
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings()
        )
        self.collection = None

    def initialize_collection(self, collection_name="documents"):
        """
        Initializes or loads a ChromaDB collection.
        
        Args:
            collection_name (str): Name of the collection.
        """
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_embeddings(self, chunks_with_embeddings: list[dict]):
        """
        Adds embeddings and metadata to the ChromaDB collection.
        
        Args:
            chunks_with_embeddings (list[dict]): List of chunks with embeddings and metadata.
        """
        if not self.collection:
            raise ValueError("Collection is not initialized. Call `initialize_collection` first.")

        # Prepare data for insertion
        ids = [chunk["id"] for chunk in chunks_with_embeddings]
        embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
        metadatas = [chunk["metadata"] for chunk in chunks_with_embeddings]
        documents = [chunk["content"] for chunk in chunks_with_embeddings]

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_embedding: list, top_k: int = 3):
        """
        Queries the ChromaDB collection for the most similar documents.
        
        Args:
            query_embedding (list): Embedding vector for the query text.
            top_k (int): Number of top results to retrieve.
        
        Returns:
            list[dict]: List of results with metadata and similarity scores.
        """
        if not self.collection:
            raise ValueError("Collection is not initialized. Call `initialize_collection` first.")

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format the results to include metadata
        formatted_results = [
            {
                "content": results["documents"][i][0],
                "metadata": results["metadatas"][i][0],
                "score": results["distances"][i][0]
            }
            for i in range(len(results["ids"]))
        ]

        return formatted_results
    