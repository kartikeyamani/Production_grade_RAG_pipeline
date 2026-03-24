import os
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, VectorStoreConfig, RAGEngineConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        try:
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)

            # Create root artifacts folder
            create_directories([self.config.artifacts_root])
        except Exception as e:
            import sys
            from src.exception.custom_exception import CustomException
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.data_ingestion
            params = self.params

            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                data_dir=config.data_dir,
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap
            )

            return data_ingestion_config
        except Exception as e:
            import sys
            from src.exception.custom_exception import CustomException
            raise CustomException(e, sys)

    def get_vector_store_config(self) -> VectorStoreConfig:
        try:
            config = self.config.vector_store
            params = self.params

            create_directories([config.root_dir])

            vector_store_config = VectorStoreConfig(
                root_dir=config.root_dir,
                db_dir=config.db_dir,
                raw_chunks_path=config.raw_chunks_path,
                ollama_embedding_model=params.ollama_embedding_model
            )

            return vector_store_config
        except Exception as e:
            import sys
            from src.exception.custom_exception import CustomException
            raise CustomException(e, sys)

    def get_rag_engine_config(self) -> RAGEngineConfig:
        try:
            config = self.config.rag_engine
            vector_config = self.config.vector_store
            params = self.params

            create_directories([config.root_dir])

            rag_engine_config = RAGEngineConfig(
                root_dir=config.root_dir,
                db_dir=vector_config.db_dir, # Engine needs DB to read from
                raw_chunks_path=vector_config.raw_chunks_path,
                top_k_vector=params.top_k_vector,
                top_k_bm25=params.top_k_bm25,
                ensemble_weights=params.ensemble_weights,
                flashrank_top_n=params.flashrank_top_n,
                groq_model=params.groq_model,
                llm_temperature=params.llm_temperature
            )
            
            return rag_engine_config
        except Exception as e:
            import sys
            from src.exception.custom_exception import CustomException
            raise CustomException(e, sys)
