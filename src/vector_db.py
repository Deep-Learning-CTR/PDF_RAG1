import chromadb
import faiss
import numpy as np
import pickle

class VectorDB:
    """Base class for vector databases"""
    def add(self, embeddings, documents, metadatas, ids):
        raise NotImplementedError
    
    def query(self, query_embeddings, n_results):
        raise NotImplementedError
    
    def count(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class ChromaDBStore(VectorDB):
    """ChromaDB implementation"""
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("document_collection")
    
    def add(self, embeddings, documents, metadatas, ids):
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embeddings, n_results):
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return results
    
    def count(self):
        return self.collection.count()
    
    def reset(self):
        self.client.delete_collection("document_collection")
        self.collection = self.client.create_collection("document_collection")

class FAISSStore(VectorDB):
    """FAISS implementation"""
    def __init__(self):
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.dimension = None
    
    def add(self, embeddings, documents, metadatas, ids):
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings_array)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_embeddings, n_results):
        if self.index is None or self.index.ntotal == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        query_array = np.array(query_embeddings).astype('float32')
        distances, indices = self.index.search(query_array, min(n_results, self.index.ntotal))
        
        results = {
            "documents": [[self.documents[i] for i in indices[0]]],
            "metadatas": [[self.metadatas[i] for i in indices[0]]],
            "distances": [distances[0].tolist()]
        }
        return results
    
    def count(self):
        return self.index.ntotal if self.index else 0
    
    def reset(self):
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.dimension = None
    
    def save(self, filepath="faiss_store.pkl"):
        """Save FAISS index and metadata to disk"""
        if self.index:
            faiss.write_index(self.index, f"{filepath}.index")
            with open(f"{filepath}.meta", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'ids': self.ids,
                    'dimension': self.dimension
                }, f)
    
    def load(self, filepath="faiss_store.pkl"):
        """Load FAISS index and metadata from disk"""
        try:
            self.index = faiss.read_index(f"{filepath}.index")
            with open(f"{filepath}.meta", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
                self.ids = data['ids']
                self.dimension = data['dimension']
            return True
        except:
            return False
