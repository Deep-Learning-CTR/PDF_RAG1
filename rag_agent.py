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
        system_prompt = "You are a helpful assistant. Use context from the PDF to answer questions. If insufficient context, say so."

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

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = os.path.join("temp.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully! Processing...")

    documents = extract_text_from_pdf(pdf_path)
    chunks = split_chunk_overlap(documents)
    texts, embeddings = embed_chunks(chunks)

    if collection.count() == 0:
        store_in_chromadb(collection, chunks, embeddings)
        st.info(f"Stored {len(chunks)} chunks in ChromaDB.")
    else:
        st.info("Using existing ChromaDB collection.")

    # Chat Interface
    st.subheader("Ask questions about the document")
    query = st.text_input("Your Question")

    if query:
        with st.spinner("Generating answer..."):
            answer, retrieved_chunks, metadatas = generate_answer_gemini(query)

        st.markdown("### 🤖 Answer")
        st.write(answer)

        with st.expander("📖 Retrieved Context"):
            for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadatas)):
                st.markdown(f"**Context {i+1} (Page {meta['page']})**")
                st.write(chunk[:500] + ("..." if len(chunk) > 500 else ""))
