import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_04_evaluation import EvaluationPipeline
from src.config.configuration import ConfigurationManager
from src.components.rag_engine import RAGEngine
from src.logger.custom_logger import logger

def ingest():
    try:
        logger.info("Starting Data Ingestion Pipeline...")
        pipeline = DataIngestionPipeline()
        pipeline.main()
        logger.info("Ingestion complete. You can now run chat mode.")
    except Exception as e:
        logger.exception(e)
        raise e

def evaluate():
    try:
        logger.info("Starting RAGAS Evaluation Pipeline...")
        pipeline = EvaluationPipeline()
        pipeline.main()
        logger.info("Evaluation complete. Results saved to artifacts/evaluation/")
    except Exception as e:
        logger.exception(e)
        raise e

def chat():
    logger.info("Starting Chat Mode...")
    config_manager = ConfigurationManager()
    rag_config = config_manager.get_rag_engine_config()
    
    rag_engine = RAGEngine(config=rag_config)
    rag_chain = rag_engine.setup_rag_pipeline()
    
    if not rag_chain:
        logger.error("Failed to setup RAG pipeline. Did you run ingestion first?")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("RAG Chat Ready! (type 'exit' or 'quit' to end)")
    print("="*50 + "\n")
    
    session_id = "default_session"
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
                
            print("\nThinking...")
            response = rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            print(f"\nAssistant: {response['answer']}")
            
            # Print sources inline
            if 'context' in response and response['context']:
                print("\nSources:")
                for i, doc in enumerate(response['context'][:3]):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    print(f"  [{i+1}] {source} (Page {page})")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Grade RAG Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from data/ and build vector store")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat session")
    parser.add_argument("--evaluate", action="store_true", help="Generate synthetic testset and score the RAG pipeline with ragas")
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest()
    elif args.chat:
        chat()
    elif args.evaluate:
        evaluate()
    else:
        parser.print_help()
