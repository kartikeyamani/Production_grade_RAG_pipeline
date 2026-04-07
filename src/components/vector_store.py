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

from langfuse import observe

class VectorStore:
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        
    def get_embeddings(self):
        """Returns the OpenAI embeddings model instance."""
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = os.getenv("OPENAI_API_KEY")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )
        except Exception as e:
            raise CustomException(e, sys)

    @observe(name="vector_db_embedding_creation")
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

            logger.info(f"Embedding {len(chunks)} chunks directly into ChromaDB...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(db_dir)
            )

            # Save plain chunks for BM25 retriever (list of Document objects)
            chunks_path = self.config.raw_chunks_path
            with open(chunks_path, "wb") as f:
                pickle.dump(chunks, f)

            logger.info("Vector database and raw chunks created successfully.")
            return vectorstore
            
        except Exception as e:
            raise CustomException(e, sys)
