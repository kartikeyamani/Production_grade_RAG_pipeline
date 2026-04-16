import streamlit as st
import os


from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.config.configuration import ConfigurationManager
from src.components.rag_engine import RAGEngine
from src.logger.custom_logger import logger

st.set_page_config(
    page_title="Production RAG",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    /* Vibrant Title */
    .premium-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 0px;
        padding-bottom: 10px;
        animation: gradient-shift 5s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient-shift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        color: white;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        border-radius: 8px;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(78, 205, 196, 0.15);
        color: #4ECDC4;
        border: 1px solid #4ECDC4;
    }

</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="premium-title">✨ Enterprise Synapse RAG</h1>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: -15px; margin-bottom: 25px; color: #888; font-size: 1.1rem; font-weight: 500;">Optimized with a 2000-chunk index for rich, organic contextual intelligence.</div>', unsafe_allow_html=True)

# --- Initialize Config & Engine ---
@st.cache_resource
def get_rag_chain():
    try:
        config_manager = ConfigurationManager()
        # Get the rag engine details (will use params.yaml chunk=2000)
        rag_config = config_manager.get_rag_engine_config()
        rag_engine = RAGEngine(config=rag_config)
        return rag_engine.setup_rag_pipeline()
    except Exception as e:
        logger.error(f"Error initializing RAG engine: {e}")
        return None

# --- Sidebar: Configuration & Data Pipeline ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("Manage the underlying semantic engine properties and data flows.")
    
    st.markdown("### System Status")
    if "rag_chain" in st.session_state and st.session_state.rag_chain:
        st.markdown('<div class="status-badge">🟢 RAG Online (2000 Chunk)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge" style="background: rgba(255, 107, 107, 0.15); color: #FF6B6B; border-color: #FF6B6B;">🔴 RAG Offline</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Data Pipeline")
    st.write("Trigger the modular ETL pipeline to parse and index documents.")
    
    if st.button("Run Data Ingestion"):
        with st.status("Initializing Knowledge Ingestion...", expanded=True) as status:
            try:
                st.write("Loading modular components...")
                pipeline = DataIngestionPipeline()
                st.write("Processing documents & chunking (Size: 2000)...")
                pipeline.main()
                status.update(label="Ingestion complete!", state="complete", expanded=False)
                st.success("Vector Store Synchronized!", icon="✅")
                # Clear session state cache if new data is ingested
                st.cache_resource.clear()
                
                # Refresh chain 
                st.session_state.rag_chain = get_rag_chain()
                st.rerun()
                    
            except Exception as e:
                status.update(label="Ingestion failed.", state="error")
                st.error(f"Pipeline error: {e}")
                
    st.markdown("---")
    st.markdown("### Session")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Memory wiped! How can I help you using our newly optimized 2000-chunk index?"}]
        st.rerun()

# --- Initialize RAG Pipeline ---
if "rag_chain" not in st.session_state:
    with st.spinner("Waking up RAG Engine..."):
        st.session_state.rag_chain = get_rag_chain()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I'm synchronized with the 2000-chunk hyper-context index. How can I assist you today?"}]

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Context Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# React to user input
if prompt := st.chat_input("Query the knowledge base..."):
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
        with st.spinner("Traversing knowledge graph..."):
            try:
                session_id = "streamlit_session_prod" 
                
                response = st.session_state.rag_chain.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )
                
                answer = response.get('answer', "I couldn't find an answer based on the current context.")
                st.markdown(answer)
                
                # Format sources
                sources_list = []
                if 'context' in response and response['context']:
                    with st.expander("🔍 Verified Context Sources", expanded=True):
                        for i, doc in enumerate(response['context'][:3]):
                            source_meta = doc.metadata.get('source', 'Unknown Document')
                            page_meta = doc.metadata.get('page', 'N/A')
                            # clean path
                            source_name = os.path.basename(str(source_meta))
                            source_text = f"**{source_name}** (Page {page_meta})"
                            st.write(f"{i+1}. {source_text}")
                            sources_list.append(source_text)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_list
                })
                
            except Exception as e:
                error_msg = f"Inference engine encountered an error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
