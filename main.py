import argparse
import sys
from src.document_processor import load_and_chunk_documents
from src.vector_store import create_vector_store
from src.rag_engine import setup_rag_pipeline

def ingest():
    print("Starting ingestion process...")
    chunks = load_and_chunk_documents()
    if not chunks:
        print("Ingestion failed: No chunks generated.")
        sys.exit(1)
        
    create_vector_store(chunks)
    print("Ingestion complete. You can now run chat mode.")

def chat():
    print("Starting chat mode...")
    rag_chain = setup_rag_pipeline()
    if not rag_chain:
        print("Failed to setup RAG pipeline. Did you run ingestion first?")
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
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Grade RAG - Phase 1")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from data/ and build vector store")
    parser.add_argument("--chat", action="store_true", help="Start the interactive chat session")
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest()
    elif args.chat:
        chat()
    else:
        parser.print_help()
