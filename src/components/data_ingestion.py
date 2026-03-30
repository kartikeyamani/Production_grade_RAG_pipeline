import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataIngestionConfig
import sys

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """
        Loads PDFs from the data directory and splits them into chunks.
        """
        try:
            data_dir = self.config.data_dir
            
            if not os.path.exists(data_dir):
                logger.warning(f"Directory {data_dir} does not exist. Creating it.")
                os.makedirs(data_dir, exist_ok=True)
                return []
            
            # Load all PDFs from directory
            logger.info(f"Loading PDFs from {data_dir}...")
            loader = PyPDFDirectoryLoader(data_dir)
            docs = loader.load()
            
            if not docs:
                logger.warning(f"No documents found in {data_dir}. Please add PDF files.")
                return []
                
            logger.info(f"Loaded {len(docs)} document pages.")
            
            # Split text into PARENT chunks (Large context blocks)
            logger.info("Splitting text into large PARENT chunks for ParentDocumentRetrieval...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Successfully generated {len(chunks)} Parent chunks.")
            
            return chunks
            
        except Exception as e:
            raise CustomException(e, sys)
