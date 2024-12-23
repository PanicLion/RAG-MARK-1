from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(filepath: str = None, file_bytes: bytes = None, password: str = None, extract_images: bool = False):
    """
    Loads content from a PDF file or byte stream.

    Args:
        filepath (str, optional): Path to the PDF file.
        file_bytes (bytes, optional): PDF content as a byte stream.
        password (str, optional): Password for encrypted PDFs. Defaults to None.
        extract_images (bool, optional): Whether to extract images. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries, each containing document metadata and content.
    """
    try:
        if file_bytes:
            # Use BytesIO for byte stream input
            file_obj = BytesIO(file_bytes)
            loader = PyPDFLoader(file_obj, password, extract_images)
        else:
            # Filepath-based loading
            loader = PyPDFLoader(filepath, password, extract_images)

        docs_lazy = loader.lazy_load()  # Lazy loading for efficient memory use
        docs = [
            {
                "content": doc.page_content,  # Extracted text
                "metadata": doc.metadata     # Metadata (e.g., page number, source file)
            }
            for doc in docs_lazy
        ]
        return docs

    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the PDF: {e}")
