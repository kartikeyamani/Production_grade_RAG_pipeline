import os
import shutil
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.config import DB_DIR, OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL

def get_embeddings():
    """Returns the Ollama embeddings model instance."""
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

def create_vector_store(chunks, db_dir=DB_DIR):
    """
    Creates and persists a Chroma vector database from document chunks.
    Clears out the old database if it exists.
    """
    if os.path.exists(db_dir):
        print(f"Clearing old vector database at {db_dir}...")
        try:
            import time
            # Sometimes Windows needs a slight delay if a process just closed
            time.sleep(1)
            shutil.rmtree(db_dir)
        except Exception as e:
            print(f"\nWarning: Could not clear the old database directory due to an OS lock: {e}")
            print("This usually happens if:")
            print("1. OneDrive is currently syncing the 'db' folder.")
            print("2. Another Python script or terminal is still using the database.")
            print("We will continue by appending to the existing database instead.\n")

    os.makedirs(db_dir, exist_ok=True)
    
    print("Initializing embeddings model...")
    embeddings = get_embeddings()
    
    print(f"Embedding {len(chunks)} chunks and storing in Chroma...")
    # Chroma automatically persists to the persist_directory when used like this
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    # Save the raw chunks to disk so we can load them into BM25 later
    chunks_path = os.path.join(db_dir, "raw_chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
        
    print("Vector database and raw chunks created successfully.")
    return vectorstore

def get_vector_store(db_dir=DB_DIR):
    """
    Loads an existing Chroma vector database.
    """
    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        print("Vector database not found. Please ingest documents first.")
        return None
        
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    return vectorstore

def get_raw_chunks(db_dir=DB_DIR):
    """
    Loads the raw text chunks required for BM25.
    """
    chunks_path = os.path.join(db_dir, "raw_chunks.pkl")
    if not os.path.exists(chunks_path):
        return None
        
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return chunks
