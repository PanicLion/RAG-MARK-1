from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_into_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into token-aware chunks while preserving metadata.

    Args:
        docs (list[dict]): List of documents with 'content' and 'metadata'.
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between chunks.

    Returns:
        list[dict]: List of chunks with 'content' and 'metadata'.
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split each document into chunks
    chunks = []
    for doc in docs:
        # Split the content into chunks
        splits = text_splitter.split_text(doc["content"])
        
        # Associate each chunk with the document's metadata
        for chunk in splits:
            chunks.append({
                "content": chunk,
                "metadata": doc["metadata"]
            })
    
    return chunks
