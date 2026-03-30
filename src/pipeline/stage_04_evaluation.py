import sys
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()

            # Get configs needed for both evaluation and the live RAG engine
            eval_config = config.get_evaluation_config()
            rag_config = config.get_rag_engine_config()

            evaluator = ModelEvaluation(eval_config=eval_config, rag_config=rag_config)

            # Stage 1: Generate synthetic testset from existing document chunks
            logger.info("Stage 1/2: Generating synthetic testset...")
            evaluator.generate_testset()

            # Stage 2: Run evaluation against the live pipeline
            logger.info("Stage 2/2: Scoring RAG pipeline with ragas metrics...")
            evaluator.evaluate()

            logger.info("Evaluation pipeline completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)
