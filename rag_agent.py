import streamlit as st
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
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
def load_embedding_model():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    return SentenceTransformer(model_name)

model = load_embedding_model()

# Initialize Gemini client
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

def rag_pipeline(user_query, system_prompt=None):
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Please use context from the PDF to answer questions. If insufficient context, say so."

    results = search_query(user_query)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])

    prompt = f"""{system_prompt}

Context:
{context}

User Question: {user_query}

Answer:"""
    return prompt, chunks, metadatas

def generate_answer_gemini(user_query):
    prompt, chunks, metadatas = rag_pipeline(user_query)
    response = llm.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"temperature": 0}
    )
    return response.text, chunks, metadatas

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("📄 RAG PDF Assistant with Gemini + ChromaDB")

# Sidebar for configuration
st.sidebar.header("⚙️ RAG Settings")
chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
top_k = st.sidebar.slider("Top K (retrieved chunks)", min_value=1, max_value=10, value=3, step=1)
# temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0, step=0.05)
# Add reprocess button
reprocess = st.sidebar.button("🔄 Reprocess PDF")

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

    # Check if first upload or reprocess is requested
    if reprocess or collection.count() == 0:
        st.info(f"Processing {len(uploaded_files)} PDF files with current chunk settings...")

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

        st.success(f"Stored {len(chunks)} chunks from {len(uploaded_files)} PDF files with chunk_size={chunk_size}, overlap={chunk_overlap}.")
    else:
        st.info("Using existing ChromaDB collection.")
    # Chat Interface
    st.subheader("Ask questions about the documents")
    query = st.text_input("Your Question")

    if query:
        with st.spinner("Generating answer..."):
            def search_query(query, n_results=3):
                query_embedding = model.encode([query])
                results = collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results
                )
                return results

            def rag_pipeline(user_query, system_prompt=None):
                if system_prompt is None:
                    system_prompt = "You are a helpful assistant.Please use context from the PDF to answer questions. If insufficient context, say so."

                results = search_query(user_query, n_results=top_k)
                chunks = results["documents"][0]
                metadatas = results["metadatas"][0]
                scores = results["distances"][0]  # similarity scores

                context = "\n\n".join(
                    [f"Context {i+1} (Score: {scores[i]:.4f}):\n{chunk}" 
                     for i, chunk in enumerate(chunks)]
                )

                prompt = f"""{system_prompt}

Context:
{context}

User Question: {user_query}

Answer:"""
                return prompt, chunks, metadatas, scores

            def generate_answer_gemini(user_query):
                prompt, chunks, metadatas, scores = rag_pipeline(user_query)
                response = llm.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config={"temperature": 0}
                )
                return response.text, chunks, metadatas, scores

            answer, retrieved_chunks, metadatas, scores = generate_answer_gemini(query)

        st.markdown("### 🤖 Answer")
        st.write(answer)

        with st.expander("📖 Retrieved Context"):
            for i, (chunk, meta, score) in enumerate(zip(retrieved_chunks, metadatas, scores)):
                filename = meta.get('filename', 'unknown')
                st.markdown(f"**Context {i+1} ({filename}, Page {meta['page']}, Score: {score:.4f})**")
                st.write(chunk)  # show full chunk without truncation
