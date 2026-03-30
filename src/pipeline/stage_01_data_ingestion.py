from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.vector_store import VectorStore
from src.logger.custom_logger import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            
            # 1. Chunk PDFs
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            chunks = data_ingestion.initiate_data_ingestion()
            
            # 2. Build ChromaDB
            if chunks:
                vector_store_config = config.get_vector_store_config()
                vector_store = VectorStore(config=vector_store_config)
                vector_store.initiate_vector_store(chunks)
            else:
                logger.warning("No chunks generated. Skipping vector store creation.")
        except Exception as e:
            import sys
            from src.exception.custom_exception import CustomException
            raise CustomException(e, sys)
