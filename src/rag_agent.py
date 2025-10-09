import streamlit as st
import os
import datetime
import pandas as pd
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from groq import Groq
from langchain_ollama import OllamaLLM as Ollama

from vector_db import VectorDB, ChromaDBStore, FAISSStore
from extractors import (
    extract_text_from_multiple_files,
    extract_text_from_pdf_advanced,
    extract_text_from_excel,
    extract_from_standalone_image,
    split_chunk_overlap
)

load_dotenv()

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="RAG Document Assistant", layout="wide")

# Initialize embedding model (cached)
@st.cache_resource
def load_embedding_model(model_name):
    if model_name == 'nomic-ai/nomic-embed-text-v1.5':
        return SentenceTransformer(model_name, trust_remote_code=True)
    else:
        return SentenceTransformer(model_name)

# Initialize LLM clients
llm2=Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
llm = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
llm3 = Ollama(model="phi3:latest")

def embed_chunks(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)
    return texts, embeddings

def store_in_vector_db(vector_db, chunks, embeddings):
    texts = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": chunk.metadata.get("source", "unknown"),
            "filename": chunk.metadata.get("filename", "unknown"),
            "file_type": chunk.metadata.get("file_type", "unknown"),
            "page": chunk.metadata.get("page", 0),
            "sheet": chunk.metadata.get("sheet", "N/A"),
            "extraction_method": chunk.metadata.get("extraction_method", "N/A"),
            "chunk_index": i
        }
        for i, chunk in enumerate(chunks)
    ]
    vector_db.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

def search_query(query, n_results=3):
    query_embedding = model.encode([query])
    results = vector_db.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results

def rag_pipeline(user_query, system_prompt=None, top_k=3):
    if system_prompt is None:
        system_prompt = """You are a helpful assistant. Please use context from the documents to answer questions. 
        Pay special attention to tables and structured data marked with [TABLE] and [END TABLE] tags.
        When referencing data from tables, be precise and cite the specific rows or columns.
        If insufficient context, say so."""

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

def generate_answer(user_query, provider="cerebras", model_name="llama-4-scout-17b-16e-instruct", top_k=3):
    prompt, chunks, metadatas, scores = rag_pipeline(user_query, top_k=top_k)

    if provider.lower() == "cerebras":
        response = llm.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content
    elif provider.lower() == "groq":
        response = llm2.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content
    else:  # Ollama
        llm3.model = model_name
        llm3.temperature = 0
        answer = llm3.invoke(prompt)

    return answer, chunks, metadatas, scores

# -------------------------------
# Chat Management Functions
# -------------------------------
def add_message_to_history(role, content, retrieved_chunks=None, metadatas=None, scores=None, response_time=None):
    """Add a message to chat history with metadata"""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "retrieved_chunks": retrieved_chunks or [],
        "metadatas": metadatas or [],
        "scores": scores or [],
        "response_time": response_time
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
            # Show timestamp and response time if available
            if message.get("response_time"):
                st.caption(f"🕒 {message['timestamp']} | ⏱️ Response time: {message['response_time']:.2f}s")
            else:
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
                        file_type = meta.get('file_type', 'unknown')
                        extraction_method = meta.get('extraction_method', 'N/A')
                        
                        # Display metadata based on file type
                        if file_type == 'pdf':
                            location = f"Page {meta.get('page', 'N/A')}"
                            method_info = f" ({extraction_method})" if extraction_method != 'N/A' else ""
                        elif file_type == 'excel':
                            location = f"Sheet: {meta.get('sheet', 'N/A')}"
                            method_info = ""
                        else:
                            location = "N/A"
                            method_info = ""
                        
                        st.markdown(f"**Context {i+1} ({filename}, {location}{method_info}, Score: {score:.4f})**")
                        st.text(chunk)

# -------------------------------
# Session State Initialization
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "current_embedding_model" not in st.session_state:
    st.session_state.current_embedding_model = None

if "current_vector_db" not in st.session_state:
    st.session_state.current_vector_db = None

if "vector_db_instance" not in st.session_state:
    st.session_state.vector_db_instance = None

if "system_notifications" not in st.session_state:
    st.session_state.system_notifications = {
        "upload_info": None,
        "db_info": None
    }

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("📄 RAG Document Assistant with Advanced PDF Processing")
st.caption("Supports PDF (with tables & complex structures) and Excel files")

# Sidebar for configuration
st.sidebar.header("⚙️ RAG Settings")

# Vector Database selection
vector_db_option = st.sidebar.selectbox(
    "Vector Database",
    ["ChromaDB", "FAISS"],
    index=0
)

# Initialize or switch vector database (persistent across reruns)
if "vector_db_instance" not in st.session_state or st.session_state.current_vector_db != vector_db_option:
    st.session_state.current_vector_db = vector_db_option
    if vector_db_option == "ChromaDB":
        st.session_state.vector_db_instance = ChromaDBStore()
    else:
        st.session_state.vector_db_instance = FAISSStore()
    st.session_state.docs_embedded = False  # force re-embed on DB switch
    st.sidebar.success(f"Switched to {vector_db_option}")

vector_db = st.session_state.vector_db_instance
st.sidebar.info(f"🗄️ Current Vector DB: {vector_db_option}")

# Embedding model selection
embedding_model_option = st.sidebar.selectbox(
    "Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "nomic-ai/nomic-embed-text-v1.5"],
    index=1
)

# Load the selected embedding model
model = load_embedding_model(embedding_model_option)
st.sidebar.info(f"📊 Current embedding model: {embedding_model_option}")

# LLM API Provider selection
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Cerebras", "Groq", "Ollama"],
    index=0
)

# LLM model selection based on provider
if llm_provider == "Cerebras":
    llm_models = {
        "Llama 4 Scout (17B)": "llama-4-scout-17b-16e-instruct",
        "Llama 3.1 8B": "llama3.1-8b",
        "Llama 3.3 70B": "llama-3.3-70b",
        "OpenAI GPT OSS (120B)": "gpt-oss-120b",
        "Qwen 3 32B": "qwen-3-32b"
    }
elif llm_provider == "Groq":
    llm_models = {
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "Llama 3.1 70B": "llama-3.1-70b-versatile",
        "Gemma 2 9B": "gemma2-9b-it"
    }
else:  # Ollama
    llm_models = {
        "Phi-3": "phi3:latest",
        "GPT-OSS 20B": "gpt-oss:20b",
        "DeepSeek R1 8B": "deepseek-r1:8b",
        "Qwen 3 8B": "qwen3:8b",
        "Gemma 3 12B": "gemma3:12b"
    }

llm_model_display = st.sidebar.selectbox(
    "LLM Model",
    list(llm_models.keys()),
    index=0
)
llm_model_option = llm_models[llm_model_display]
st.sidebar.info(f"🤖 Provider: {llm_provider} | Model: {llm_model_display}")

chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
top_k = st.sidebar.slider("Top K (retrieved chunks)", min_value=1, max_value=10, value=3, step=1)

# Vision model toggle
use_vision = st.sidebar.checkbox("Use Vision Model for Images (Groq)", value=True, help="Extract context from non-text images using Groq's vision model")

# Add reprocess button
reprocess = st.sidebar.button("🔄 Reprocess Documents")

# Chat management
st.sidebar.header("💬 Chat Management")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

if "docs_embedded" not in st.session_state:
    st.session_state.docs_embedded = False

if st.session_state.chat_history:
    st.sidebar.write(f"Messages: {len(st.session_state.chat_history)}")

# Info box about PDF processing
with st.sidebar.expander("ℹ️ PDF Processing Info"):
    st.write("""
    **Advanced PDF Features:**
    - Table extraction using Camelot & PDFPlumber
    - OCR for text extraction from images
    - Vision AI for non-text image understanding
    - Layout preservation
    - Complex structure handling
    - Automatic fallback mechanisms

    **Image Processing:**
    - First tries OCR to extract text from images
    - If no text found, uses Groq vision model to describe image content
    - Supports charts, diagrams, photos, and visual elements

    Tables are marked with [TABLE] tags, images with [IMAGE DESCRIPTION] tags.
    """)

with st.sidebar.expander("ℹ️ Vector Database Info"):
    st.write("""
    **ChromaDB:**
    - In-memory storage
    - Built-in metadata filtering
    - Easy to use
    
    **FAISS:**
    - High-performance similarity search
    - Better for large datasets (>100k vectors)
    - Requires more memory
    - Supports saving/loading indices
    """)
    
    if vector_db_option == "FAISS":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Index"):
                if isinstance(vector_db, FAISSStore) and vector_db.count() > 0:
                    vector_db.save()
                    st.success("Index saved!")
                else:
                    st.warning("No data to save")
        with col2:
            if st.button("📂 Load Index"):
                if isinstance(vector_db, FAISSStore):
                    if vector_db.load():
                        st.success("Index loaded!")
                    else:
                        st.error("Failed to load index")

# File uploader - accepts documents and images
uploaded_files = st.file_uploader(
    "Upload documents and images",
    type=["pdf", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("💡 Images will be analyzed using Vision AI. PDFs may contain images that will also be analyzed.")

    # Separate files into documents and images
    doc_files = []
    image_files = []

    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext in ['.png', '.jpg', '.jpeg', '.webp']:
            image_files.append(uploaded_file)
        else:
            doc_files.append(uploaded_file)

    # Store upload info
    upload_info_parts = []
    if doc_files:
        upload_info_parts.append(f"{len(doc_files)} document(s)")
    if image_files:
        upload_info_parts.append(f"{len(image_files)} image(s)")
    st.session_state.system_notifications["upload_info"] = f"{' + '.join(upload_info_parts)} uploaded successfully!"

    # Check if model changed since last run
    model_changed = st.session_state.current_embedding_model != embedding_model_option
    if model_changed:
        st.session_state.current_embedding_model = embedding_model_option

    # Determine whether to reprocess
    should_reset = (
        reprocess
        or vector_db.count() == 0
        or model_changed
        or not st.session_state.docs_embedded
    )

    if should_reset:
        if model_changed:
            st.info(f"Embedding model changed to {embedding_model_option}. Reprocessing all files...")
        elif reprocess:
            st.info("🔄 Reprocessing all documents as requested...")
        elif vector_db.count() == 0:
            st.info("Processing new documents...")
        else:
            st.info("Processing uploaded documents for the first time...")

        # Reset DB and mark as not embedded
        vector_db.reset()
        st.session_state.docs_embedded = False

        all_documents = []

        # -------------------------------
        # Process document files (PDF, Excel)
        # -------------------------------
        if doc_files:
            file_paths = []
            for i, uploaded_file in enumerate(doc_files):
                file_path = os.path.join(f"temp_{i}_{uploaded_file.name}")
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                file_paths.append(file_path)

            with st.spinner("Extracting text, tables, and images from documents..."):
                documents = extract_text_from_multiple_files(file_paths, use_vision)
                all_documents.extend(documents)

        # -------------------------------
        # Process standalone image files
        # -------------------------------
        if image_files:
            image_paths = []
            for i, uploaded_image in enumerate(image_files):
                image_path = os.path.join(f"temp_img_{i}_{uploaded_image.name}")
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.read())
                image_paths.append(image_path)

            with st.spinner("Analyzing images with vision model..."):
                for img_path in image_paths:
                    docs = extract_from_standalone_image(img_path)
                    all_documents.extend(docs)

        # -------------------------------
        # Chunk, embed, and store
        # -------------------------------
        if all_documents:
            with st.spinner("Chunking documents..."):
                chunks = split_chunk_overlap(all_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            with st.spinner("Generating embeddings..."):
                texts, embeddings = embed_chunks(chunks)

            with st.spinner(f"Storing in {vector_db_option}..."):
                store_in_vector_db(vector_db, chunks, embeddings)

            success_msg = f"✅ Stored {len(chunks)} chunks from {len(all_documents)} items in {vector_db_option}"
            st.session_state.system_notifications["db_info"] = success_msg
            st.session_state.docs_embedded = True  # ✅ mark as processed
            st.success(success_msg)
        else:
            st.warning("No content could be extracted from the uploaded files.")

    else:
        st.info("✅ Documents already embedded. Ready for chatting!")

# Display persistent system notifications (above chat interface)
if st.session_state.system_notifications["upload_info"]:
    st.success(st.session_state.system_notifications["upload_info"])
if st.session_state.system_notifications["db_info"]:
    st.info(st.session_state.system_notifications["db_info"])

# Chat Interface - moved outside uploaded_files block
if vector_db.count() > 0:
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
                start_time = time.time()
                answer, retrieved_chunks, metadatas, scores = generate_answer(query, llm_provider, llm_model_option, top_k)
                response_time = time.time() - start_time

            # Display assistant response
            st.write(answer)
            st.caption(f"🕒 {datetime.datetime.now().strftime('%H:%M:%S')} | ⏱️ Response time: {response_time:.2f}s")

            # Show retrieved context
            with st.expander("📖 Retrieved Context", expanded=False):
                for i, (chunk, meta, score) in enumerate(zip(retrieved_chunks, metadatas, scores)):
                    filename = meta.get('filename', 'unknown')
                    file_type = meta.get('file_type', 'unknown')
                    extraction_method = meta.get('extraction_method', 'N/A')

                    # Display metadata based on file type
                    if file_type == 'pdf':
                        location = f"Page {meta.get('page', 'N/A')}"
                        method_info = f" ({extraction_method})" if extraction_method != 'N/A' else ""
                    elif file_type == 'excel':
                        location = f"Sheet: {meta.get('sheet', 'N/A')}"
                        method_info = ""
                    else:
                        location = "N/A"
                        method_info = ""

                    st.markdown(f"**Context {i+1} ({filename}, {location}{method_info}, Score: {score:.4f})**")
                    st.text(chunk)

            # Add assistant message to history
            add_message_to_history("assistant", answer, retrieved_chunks, metadatas, scores, response_time)
else:
    st.info("👆 Please upload documents to start chatting")