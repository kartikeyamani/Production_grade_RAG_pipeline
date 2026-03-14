from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from src.config import GROQ_MODEL, GROQ_API_KEY
from src.vector_store import get_vector_store, get_raw_chunks

# In-memory session tracking for simplicity
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def setup_rag_pipeline():
    """
    Sets up the Conversational RAG pipeline using LangChain LCEL.
    """
    vectorstore = get_vector_store()
    raw_chunks = get_raw_chunks()
    
    if not vectorstore or not raw_chunks:
        print("Vector database or raw chunks not found.")
        return None
        
    # Semantic Search
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Keyword Search (Sparse)
    bm25_retriever = BM25Retriever.from_documents(raw_chunks)
    bm25_retriever.k = 10
    
    # Hybrid Search (Merges top 10 semantic + top 10 keyword using Reciprocal Rank Fusion)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )
    
    # Reranker (Squashes the 20 results down to the 3 absolute best matches)
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    # Check if Groq API key is present
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return None

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0
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
    
    return conversational_rag_chain
