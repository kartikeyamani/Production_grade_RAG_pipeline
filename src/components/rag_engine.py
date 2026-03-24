import os
import sys
import pickle
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import RAGEngineConfig

# In-memory session tracking for simplicity
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class RAGEngine:
    def __init__(self, config: RAGEngineConfig):
        self.config = config

    def _get_vector_store(self):
        """Loads an existing Chroma vector database."""
        try:
            db_dir = self.config.db_dir
            if not os.path.exists(db_dir) or not os.listdir(db_dir):
                logger.error("Vector database not found. Please ingest documents first.")
                return None
                
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Ensure we use same embedding model that vector store uses
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",  # Ideally read from config, but hardcoding for simplicity given prior arch
                base_url=base_url
            )
            return Chroma(
                persist_directory=str(db_dir),
                embedding_function=embeddings
            )
        except Exception as e:
            raise CustomException(e, sys)

    def _get_raw_chunks(self):
        """Loads the raw text chunks required for BM25."""
        try:
            chunks_path = self.config.raw_chunks_path
            if not os.path.exists(chunks_path):
                logger.error("Raw chunks not found. Please ingest documents first.")
                return None
                
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            return chunks
        except Exception as e:
            raise CustomException(e, sys)

    def setup_rag_pipeline(self):
        """
        Sets up the Conversational RAG pipeline using LangChain LCEL.
        """
        try:
            vectorstore = self._get_vector_store()
            raw_chunks = self._get_raw_chunks()
            
            if not vectorstore or not raw_chunks:
                logger.error("Prerequisites for RAG pipeline missing.")
                return None
                
            # Semantic Search
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": self.config.top_k_vector})
            
            # Keyword Search (Sparse)
            bm25_retriever = BM25Retriever.from_documents(raw_chunks)
            bm25_retriever.k = self.config.top_k_bm25
            
            # Hybrid Search
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], 
                weights=self.config.ensemble_weights
            )
            
            # Reranker
            compressor = FlashrankRerank(top_n=self.config.flashrank_top_n)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=ensemble_retriever
            )
            
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                logger.error("GROQ_API_KEY not found in environment variables.")
                return None

            llm = ChatGroq(
                model=self.config.groq_model,
                temperature=self.config.llm_temperature
            )
            
            # 1. Contextualize question prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, compression_retriever, contextualize_q_prompt
            )
            
            # 2. Answer question prompt
            qa_system_prompt = (
                "You are an assistant for question-answering tasks for research papers. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Keep the answer analytical and professional. "
                "At the end of your sentences, include citations to the source document if possible. "
                "Context: {context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # 3. Add memory wrapper
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            
            logger.info("RAG Pipeline Setup successfully.")
            return conversational_rag_chain
            
        except Exception as e:
            raise CustomException(e, sys)
