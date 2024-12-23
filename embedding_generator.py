from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding generator with a specified model.
        
        Args:
            model_name (str): Name of the embedding model.
        """
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks: list[dict], chunk_prefix="chunk") -> list[dict]:
        """
        Generates embeddings for a list of text chunks and keeps metadata intact.
        
        Args:
            chunks (list[dict]): List of chunks, each containing 'content' and 'metadata'.
            chunk_prefix (str): Prefix to use for generating unique IDs for each chunk.
        
        Returns:
            list[dict]: List of chunks with 'embedding', 'id', and 'metadata' attached.
        """
        # Extract the text from the chunks (these are the 'content' fields)
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings for the text
        embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()  # Convert to list format
        
        # Attach embeddings and generate unique ids for each chunk
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]  # Attach the embedding to the metadata
            chunk["id"] = f"{chunk_prefix}_{i+1}"  # Generate a unique id for the chunk (e.g., chunk_1, chunk_2, etc.)
        
        return chunks

    def generate_query_embedding(self, query_text: str) -> list:
        """
        Generates an embedding for a single query text.
        
        Args:
            query_text (str): The query string to embed.
        
        Returns:
            list: Embedding vector for the query text.
        """
        return self.model.encode([query_text], convert_to_tensor=False).tolist()[0]
