import streamlit as st
import os
from src.document_processor import load_and_chunk_documents
from src.vector_store import create_vector_store
from src.rag_engine import setup_rag_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Chatbot (Phase 1.5)")

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Upload PDFs to your `data/` folder, then click below to ingest them.")
    
    if st.button("Ingest Documents"):
        with st.spinner("Processing documents..."):
            try:
                chunks = load_and_chunk_documents()
                if not chunks:
                    st.error("No documents found or processed in data/ folder.", icon="🚨")
                else:
                    create_vector_store(chunks)
                    st.success(f"Successfully digested {len(chunks)} chunks!", icon="✅")
                    # Clear session state cache if new data is ingested
                    if "rag_chain" in st.session_state:
                         st.session_state.pop("rag_chain")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

# --- Initialize RAG Pipeline ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = setup_rag_pipeline()

# --- Chat Interface ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(source)

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if pipeline exists
    if st.session_state.rag_chain is None:
        st.error("RAG Pipeline is not initialized. Have you ingested documents yet?")
        st.stop()

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Use a fixed session ID for the Streamlit user session
                session_id = "streamlit_session_1" 
                
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
                    st.markdown("**Sources:**")
                    for i, doc in enumerate(response['context'][:3]):
                        source_meta = doc.metadata.get('source', 'Unknown')
                        page_meta = doc.metadata.get('page', 'Unknown')
                        source_text = f"[{i+1}] {source_meta} (Page {page_meta})"
                        st.write(source_text)
                        sources_list.append(source_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_list
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
