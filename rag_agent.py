from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
import chromadb
import os
import sys

def load_embedding_model():
    """Load the embedding model with progress indication"""
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers/')
    model_path = os.path.join(cache_dir, model_name.replace('/', '_'))

    if os.path.exists(model_path):
        print("✅ Loading cached embedding model...")
    else:
        print("📥 Downloading embedding model (this may take a few minutes on first run)...")
        print("💾 Model will be cached for future use")

    try:
        model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

# Setting up chroma db
client = chromadb.Client()
collection = client.create_collection("2508.05004v2")

# Load the embedding model with progress indication
print("🚀 Initializing RAG Agent...")
model = load_embedding_model()

llm=genai.Client()


def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents=loader.load()
    return documents

def split_chunk_overlap(documents):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_documents(documents)
    return chunks

def embed_model1(chunks):

    # Extract text content from LangChain document objects
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings
    embeddings = model.encode(chunk_texts)
    
    print(f"Created embeddings for {len(chunk_texts)} chunks")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings

def store_in_chromadb(collection, chunks, embeddings):
    # Extract text content from chunks
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # Create unique IDs for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Create metadata for each chunk (optional but useful)
    metadatas = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", 0),
            "chunk_index": i
        }
        metadatas.append(metadata)
    
    # Add to ChromaDB collection
    collection.add(
        embeddings=embeddings.tolist(),  # Convert numpy array to list
        documents=chunk_texts,           # The actual text chunks
        metadatas=metadatas,            # Additional info about each chunk
        ids=ids                         # Unique identifiers
    )
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    return collection

def search_query(user_query, n_results=3):
    # Convert user query to embedding
    query_embedding = model.encode([user_query])
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    
    # Extract relevant chunks
    relevant_chunks = results['documents'][0]  # Get the documents
    metadatas = results['metadatas'][0]        # Get metadata
    distances = results['distances'][0]        # Get similarity scores
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    for i, (chunk, meta, distance) in enumerate(zip(relevant_chunks, metadatas, distances)):
        print(f"Chunk {i+1} (similarity: {1-distance:.3f}):")
        print(f"Source: {meta['source']}, Page: {meta['page']}")
        print(f"Text preview: {chunk[:100]}...")
        print("-" * 50)
    
    return relevant_chunks, metadatas, distances

def rag_pipeline(user_query, system_prompt=None):
    # Default system prompt
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately. 
        If the context doesn't contain enough information to answer the question, say so clearly."""
    
    # Get relevant chunks
    relevant_chunks, metadatas, distances = search_query(user_query)
    
    # Combine chunks into context
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
    
    # Create the full prompt
    full_prompt = f"""{system_prompt}

Context from the document:
{context}

User Question: {user_query}

Answer:"""
    
    return full_prompt, relevant_chunks
def generate_answer_gemini(user_query, system_prompt=None):
    try:
        # Get the full prompt and relevant chunks
        full_prompt, relevant_chunks = rag_pipeline(user_query, system_prompt)
        
        # Generate response using Gemini
        response = llm.models.generate_content(
            model='gemini-2.5-flash',
            contents=[full_prompt],
            config={
                'temperature': 0
            }
        )
        
        answer = response.text
        return answer, relevant_chunks
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer at this time.", []

def rag_agent(user_query, system_prompt=None):
    """
    Complete RAG agent that retrieves relevant context and generates an answer
    """
    print("🤖 RAG Agent Processing...")
    print(f"Query: {user_query}")
    print("="*60)
    
    # Generate answer using RAG pipeline
    answer, relevant_chunks = generate_answer_gemini(user_query, system_prompt)
    
    # Display results
    print("📖 Retrieved Context:")
    for i, chunk in enumerate(relevant_chunks[:2]):  # Show top 2 chunks
        print(f"\n--- Context {i+1} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print("\n🎯 AI Answer:")
    print(answer)
    print("="*60)
    
    return answer, relevant_chunks


def interactive_rag():
    """
    Interactive interface for the RAG agent
    """
    print("🚀 RAG Agent Ready!")
    print("Ask questions about your PDF document.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Your Question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
            
        if user_input:
            try:
                rag_agent(user_input)
                print("\n" + "="*60 + "\n")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please enter a question.")

# Extract text
documents = extract_text_from_pdf("2508.05004v2.pdf")
chunks= split_chunk_overlap(documents)
embeddings= embed_model1(chunks)
collection = store_in_chromadb(collection, chunks, embeddings)

print(f"Collection now contains {collection.count()} items")
# Test and Run
if __name__ == "__main__":
    # Quick test
    print("Testing RAG Agent...")
    test_query = "What is the main topic of this document?"
    rag_agent(test_query)
    
    # Start interactive mode
    print("\n" + "="*60)
    interactive_rag()

