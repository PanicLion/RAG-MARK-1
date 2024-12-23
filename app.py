import streamlit as st
from vector_store import ChromaDBHandler
from embedding_generator import EmbeddingGenerator
from llm import OllamaLLMWrapper
from pdf_loader import load_pdf
from doc_splitter import split_into_chunks

# Streamlit App Configuration
st.set_page_config(page_title="SLA Chatbot", layout="wide")

# Initialize Components
vector_db = ChromaDBHandler(persist_directory="chroma_db")
vector_db.initialize_collection(collection_name="sla_documents")
embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
ollama_wrapper = OllamaLLMWrapper()

# Define Prompt Template
prompt_template = """
You are an assistant for question-answering tasks regarding SLA documents.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise:
Context: {context}
Question: {question}
Answer:
"""
chain = ollama_wrapper.create_chain(template=prompt_template)

# Streamlit UI
st.title("📜 SLA Chatbot")
st.sidebar.header("Upload PDF File")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# File Upload Section
uploaded_file = st.sidebar.file_uploader(
    "Upload a single SLA PDF file",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file:
    st.sidebar.write(f"✅ {uploaded_file.name} uploaded successfully!")

    try:
        # Load and process the PDF directly from the file object
        pdf_bytes = uploaded_file.read()  # Read the file object as bytes
        docs = load_pdf(file_bytes=pdf_bytes)  # Pass bytes directly to `load_pdf`
        chunks = split_into_chunks(docs)
        embeddings = embedding_generator.generate_embeddings(chunks)
        vector_db.add_embeddings(embeddings)
        st.sidebar.success(f"Processed and stored embeddings for {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")

# Chat Section
st.header("💬 Chat with SLA Document")

# Display Chat History
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['message']}")
    else:
        st.markdown(f"**Bot:** {chat['message']}")

# Input Text Box (Always Available at the Bottom)
st.divider()
with st.container():
    question = st.text_input(
        "Ask a question about the SLA document:",
        key="user_question",
        placeholder="Type your question here..."
    )

    if st.button("Submit Question"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "message": question})

            # Query Vector Store
            query_embedding = embedding_generator.generate_query_embedding(question)
            retrieved_docs = vector_db.query(query_embedding=query_embedding, top_k=3)
            
            if not retrieved_docs:
                bot_response = "No relevant context found in the uploaded document."
            else:
                # Combine context from retrieved documents
                context = "\n".join([doc["content"] for doc in retrieved_docs])
                bot_response = ollama_wrapper.generate_response(context=context, question=question, chain=chain)

            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "bot", "message": bot_response})

            # Scroll back to the input box
            st.experimental_rerun()
