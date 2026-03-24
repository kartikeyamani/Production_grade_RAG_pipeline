import streamlit as st
import os

from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.config.configuration import ConfigurationManager
from src.components.rag_engine import RAGEngine
from src.logger.custom_logger import logger

# --- Page Configuration ---
st.set_page_config(
    page_title="Production RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise RAG Architecture")

# --- Initialize Config & Engine ---
@st.cache_resource
def get_rag_chain():
    try:
        config_manager = ConfigurationManager()
        rag_config = config_manager.get_rag_engine_config()
        rag_engine = RAGEngine(config=rag_config)
        return rag_engine.setup_rag_pipeline()
    except Exception as e:
        logger.error(f"Error initializing RAG engine: {e}")
        return None

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("⚙️ Data Pipeline")
    st.write("Trigger the modular ETL pipeline.")
    
    if st.button("Run Ingestion Pipeline"):
        with st.spinner("Processing documents & building vector store..."):
            try:
                pipeline = DataIngestionPipeline()
                pipeline.main()
                st.success("Pipeline executed successfully!", icon="✅")
                # Clear session state cache if new data is ingested
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error during ingestion pipeline: {e}")

# --- Initialize RAG Pipeline ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_rag_chain()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(source)

# React to user input
if prompt := st.chat_input("Ask a question about the digested knowledge base..."):
    # Refresh chain if it was none initially
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = get_rag_chain()
        
    if st.session_state.rag_chain is None:
        st.error("RAG Pipeline is offline. Please run the ingestion pipeline first.")
        st.stop()

    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            try:
                session_id = "streamlit_session_prod" 
                
                response = st.session_state.rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )
                
                answer = response.get('answer', "I couldn't find an answer.")
                st.markdown(answer)
                
                # Format sources
                sources_list = []
                if 'context' in response and response['context']:
                    st.markdown("---")
                    st.markdown("**Retrieved Contexts:**")
                    for i, doc in enumerate(response['context'][:3]):
                        source_meta = doc.metadata.get('source', 'Unknown')
                        page_meta = doc.metadata.get('page', 'Unknown')
                        source_text = f"[{i+1}] {source_meta} (Page {page_meta})"
                        st.write(source_text)
                        sources_list.append(source_text)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_list
                })
                
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
