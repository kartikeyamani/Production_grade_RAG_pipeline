import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DATA_DIR

def load_and_chunk_documents(data_dir=DATA_DIR, chunk_size=1000, chunk_overlap=200):
    """
    Loads PDFs from the data directory and splits them into chunks.
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Creating it.")
        os.makedirs(data_dir, exist_ok=True)
        return []
    
    # Load all PDFs from directory
    print(f"Loading PDFs from {data_dir}...")
    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()
    
    if not docs:
        print(f"No documents found in {data_dir}. Please add PDF files.")
        return []
        
    print(f"Loaded {len(docs)} document pages.")
    
    # Split text into chunks
    print(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"Successfully split into {len(chunks)} chunks.")
    return chunks
