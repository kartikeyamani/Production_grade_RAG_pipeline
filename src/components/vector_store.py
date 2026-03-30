import os
import shutil
import pickle
import sys
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import VectorStoreConfig

class VectorStore:
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        
    def get_embeddings(self):
        """Returns the Ollama embeddings model instance."""
        try:
            # Note: We rely on the .env file for the OLLAMA_BASE_URL to keep it secure
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(
                model=self.config.ollama_embedding_model,
                base_url=base_url
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_vector_store(self, chunks):
        """
        Creates and persists a Chroma vector database from document chunks.
        Clears out the old database if it exists.
        """
        try:
            db_dir = self.config.db_dir
            if os.path.exists(db_dir):
                logger.info(f"Clearing old vector database at {db_dir}...")
                try:
                    import time
                    time.sleep(1)
                    shutil.rmtree(db_dir)
                except Exception as e:
                    logger.warning(f"Could not clear the old database directory due to an OS lock: {e}")
                    logger.warning("Continuing by appending to existing database instead.")

            os.makedirs(db_dir, exist_ok=True)
            
            logger.info("Initializing embeddings model...")
            embeddings = self.get_embeddings()
            
            from langchain_classic.retrievers import ParentDocumentRetriever
            from langchain_core.stores import InMemoryStore
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            logger.info("Initializing ParentDocumentRetriever structure...")
            store = InMemoryStore()
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=str(db_dir)
            )
            
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
            )
            
            logger.info(f"Adding {len(chunks)} parent documents and generating child embeddings...")
            retriever.add_documents(chunks, ids=None)
            
            chunks_path = self.config.raw_chunks_path
            with open(chunks_path, "wb") as f:
                pickle.dump(store.store, f)
                
            logger.info("Vector database and raw chunks created successfully.")
            return vectorstore
            
        except Exception as e:
            raise CustomException(e, sys)
