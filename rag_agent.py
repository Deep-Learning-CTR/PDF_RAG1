import streamlit as st
import os
import datetime
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="RAG PDF Assistant", layout="wide")

# Initialize ChromaDB client
client = chromadb.Client()  
collection = client.get_or_create_collection("pdf_collection")

# Initialize embedding model (cached)
@st.cache_resource
def load_embedding_model(model_name):
    if model_name == 'nomic-ai/nomic-embed-text-v1.5':
        return SentenceTransformer(model_name, trust_remote_code=True)
    else:
        return SentenceTransformer(model_name)

# Initialize Cerebras client
llm = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# -------------------------------
# Helper Functions
# -------------------------------
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def extract_text_from_multiple_pdfs(pdf_paths):
    all_documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        # Update metadata to include the original filename
        filename = os.path.basename(pdf_path)
        for doc in documents:
            doc.metadata['filename'] = filename
        all_documents.extend(documents)
    return all_documents

def split_chunk_overlap(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def embed_chunks(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)
    return texts, embeddings

def store_in_chromadb(collection, chunks, embeddings):
    texts = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": chunk.metadata.get("source", "unknown"),
         "filename": chunk.metadata.get("filename", "unknown"),
         "page": chunk.metadata.get("page", 0),
         "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

def search_query(query, n_results=3):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results

def rag_pipeline(user_query, system_prompt=None, top_k=3):
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Please use context from the PDF to answer questions. If insufficient context, say so."

    results = search_query(user_query, n_results=top_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    scores = results["distances"][0]

    # Get recent conversation for context
    conversation_context = get_recent_conversation(n=3)

    # Build retrieved context
    context = "\n\n".join([f"Context {i+1} (Score: {scores[i]:.4f}):\n{chunk}" for i, chunk in enumerate(chunks)])

    # Build prompt with conversation history
    if conversation_context:
        prompt = f"""{system_prompt}

Previous Conversation:
{conversation_context}

Retrieved Context from Documents:
{context}

Current Question: {user_query}

Answer:"""
    else:
        prompt = f"""{system_prompt}

Retrieved Context from Documents:
{context}

Question: {user_query}

Answer:"""
    return prompt, chunks, metadatas, scores

def generate_answer_cerebras(user_query, model_name="llama-4-scout-17b-16e-instruct", top_k=3):
    prompt, chunks, metadatas, scores = rag_pipeline(user_query, top_k=top_k)
    response = llm.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content, chunks, metadatas, scores

# -------------------------------
# Chat Management Functions
# -------------------------------
def add_message_to_history(role, content, retrieved_chunks=None, metadatas=None, scores=None):
    """Add a message to chat history with metadata"""
    import datetime
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "retrieved_chunks": retrieved_chunks or [],
        "metadatas": metadatas or [],
        "scores": scores or []
    }
    st.session_state.chat_history.append(message)

def get_recent_conversation(n=3):
    """Get the last n conversation exchanges for context"""
    if not st.session_state.chat_history:
        return ""

    # Get last n*2 messages (n user + n assistant pairs)
    recent_messages = st.session_state.chat_history[-(n*2):]

    conversation = []
    for msg in recent_messages:
        if msg["role"] == "user":
            conversation.append(f"User: {msg['content']}")
        else:
            conversation.append(f"Assistant: {msg['content']}")

    return "\n".join(conversation) if conversation else ""

def display_chat_message(message):
    """Display a single chat message"""
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
            st.caption(f"🕒 {message['timestamp']}")
    else:  # assistant
        with st.chat_message("assistant"):
            st.write(message["content"])
            st.caption(f"🕒 {message['timestamp']}")

            # Show retrieved context if available
            if message["retrieved_chunks"]:
                with st.expander("📖 Retrieved Context", expanded=False):
                    for i, (chunk, meta, score) in enumerate(zip(
                        message["retrieved_chunks"],
                        message["metadatas"],
                        message["scores"]
                    )):
                        filename = meta.get('filename', 'unknown')
                        st.markdown(f"**Context {i+1} ({filename}, Page {meta['page']}, Score: {score:.4f})**")
                        st.write(chunk)

# -------------------------------
# Session State Initialization
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "current_embedding_model" not in st.session_state:
    st.session_state.current_embedding_model = None

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("📄 RAG PDF Assistant with Cerebras + ChromaDB")

# Sidebar for configuration
st.sidebar.header("⚙️ RAG Settings")

# Embedding model selection
embedding_model_option = st.sidebar.selectbox(
    "Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "nomic-ai/nomic-embed-text-v1.5"],
    index=0
)

# Load the selected embedding model
model = load_embedding_model(embedding_model_option)
st.sidebar.info(f"📊 Current embedding model: {embedding_model_option}")

# LLM model selection
llm_models = {
    "Llama 4 Scout (17B)": "llama-4-scout-17b-16e-instruct",
    "Llama 3.1 8B": "llama3.1-8b",
    "Llama 3.3 70B": "llama-3.3-70b",
    "OpenAI GPT OSS (120B)": "gpt-oss-120b",
    "Qwen 3 32B": "qwen-3-32b"
}

llm_model_display = st.sidebar.selectbox(
    "LLM Model",
    list(llm_models.keys()),
    index=0
)
llm_model_option = llm_models[llm_model_display]
st.sidebar.info(f"🤖 Current LLM: {llm_model_display}")

chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
top_k = st.sidebar.slider("Top K (retrieved chunks)", min_value=1, max_value=10, value=3, step=1)
# temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0, step=0.05)
# Add reprocess button
reprocess = st.sidebar.button("🔄 Reprocess PDF")

# Chat management
st.sidebar.header("💬 Chat Management")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.chat_history:
    st.sidebar.write(f"Messages: {len(st.session_state.chat_history)}")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Save all uploaded files
    pdf_paths = []
    for i, uploaded_file in enumerate(uploaded_files):
        pdf_path = os.path.join(f"temp_{i}_{uploaded_file.name}")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        pdf_paths.append(pdf_path)

    st.success(f"{len(uploaded_files)} PDF files uploaded successfully!")

    # Check if embedding model has changed
    model_changed = st.session_state.current_embedding_model != embedding_model_option
    if model_changed:
        st.session_state.current_embedding_model = embedding_model_option

    # Check if first upload, reprocess is requested, or model changed
    if reprocess or collection.count() == 0 or model_changed:
        if model_changed:
            st.info(f"Embedding model changed to {embedding_model_option}. Reprocessing {len(uploaded_files)} PDF files...")
        else:
            st.info(f"Processing {len(uploaded_files)} PDF files with current settings...")

        # Always reset collection when reprocessing
        client.delete_collection("pdf_collection")
        collection = client.create_collection("pdf_collection")

        documents = extract_text_from_multiple_pdfs(pdf_paths)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        texts, embeddings = embed_chunks(chunks)
        store_in_chromadb(collection, chunks, embeddings)

        st.success(f"Stored {len(chunks)} chunks from {len(uploaded_files)} PDF files using {embedding_model_option} with chunk_size={chunk_size}, overlap={chunk_overlap}.")
    else:
        st.info("Using existing ChromaDB collection.")

    # Chat Interface
    st.subheader("💬 Chat with your documents")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message)

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        add_message_to_history("user", query)

        # Display user message immediately
        with st.chat_message("user"):
            st.write(query)
            st.caption(f"🕒 {st.session_state.chat_history[-1]['timestamp']}")

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                def search_query(query, n_results=3):
                    query_embedding = model.encode([query])
                    results = collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=n_results
                    )
                    return results

                def rag_pipeline(user_query, system_prompt=None):
                    if system_prompt is None:
                        system_prompt = "You are a helpful assistant. Please use context from the PDF to answer questions. If insufficient context, say so."

                    results = search_query(user_query, n_results=top_k)
                    chunks = results["documents"][0]
                    metadatas = results["metadatas"][0]
                    scores = results["distances"][0]  # similarity scores

                    # Get recent conversation for context
                    conversation_context = get_recent_conversation(n=3)

                    # Build retrieved context
                    context = "\n\n".join(
                        [f"Context {i+1} (Score: {scores[i]:.4f}):\n{chunk}"
                         for i, chunk in enumerate(chunks)]
                    )

                    # Build prompt with conversation history
                    if conversation_context:
                        prompt = f"""{system_prompt}

Previous Conversation:
{conversation_context}

Retrieved Context from Documents:
{context}

Current Question: {user_query}

Answer:"""
                    else:
                        prompt = f"""{system_prompt}

Retrieved Context from Documents:
{context}

Question: {user_query}

Answer:"""
                    return prompt, chunks, metadatas, scores

                def generate_answer_cerebras(user_query, model_name):
                    prompt, chunks, metadatas, scores = rag_pipeline(user_query)
                    response = llm.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    return response.choices[0].message.content, chunks, metadatas, scores

                answer, retrieved_chunks, metadatas, scores = generate_answer_cerebras(query, llm_model_option)

            # Display assistant response
            st.write(answer)
            st.caption(f"🕒 {datetime.datetime.now().strftime('%H:%M:%S')}")

            # Show retrieved context
            with st.expander("📖 Retrieved Context", expanded=False):
                for i, (chunk, meta, score) in enumerate(zip(retrieved_chunks, metadatas, scores)):
                    filename = meta.get('filename', 'unknown')
                    st.markdown(f"**Context {i+1} ({filename}, Page {meta['page']}, Score: {score:.4f})**")
                    st.write(chunk)

            # Add assistant message to history
            add_message_to_history("assistant", answer, retrieved_chunks, metadatas, scores)
