# Application Workflows

Here are the high-level flowcharts for how **`main.py`** (the CLI) and **`app.py`** (the Web UI) execute their respective logic. Both use the same foundational pieces, but follow slightly different orchestrations.

## 1. The `main.py` CLI Workflow

This is how the command line handles both running the ingestion and starting a chat terminal depending on the chosen flag.

```mermaid
graph TD
    %% Base Flow
    Start(User runs main.py) --> A{Parse Arguments}
    
    %% Ingestion Flow
    A -->|--ingest| DI[DataIngestionPipeline.main]
    DI --> CM1([ConfigurationManager])
    CM1 -.-> YAML[(config.yaml & params.yaml)]
    CM1 -->|Returns Config Entities| DI
    DI -->|Initializes| DI_Comp[DataIngestion Component]
    DI_Comp -->|Returns chunks| VS_Comp[VectorStore Component]
    VS_Comp -->|Creates Chroma/Pickle| DB[(Saved artifacts/db)]
    
    %% Chat Flow
    A -->|--chat| CM2([ConfigurationManager])
    CM2 -.-> YAML
    CM2 -->|Returns RAGEngineConfig| Chat[RAGEngine Component]
    Chat -->|Loads| DB
    Chat -->|Builds| Chain[Setup Hybrid Retriever & ChatGroq LLM]
    Chain -->|Returns RAG Chain| Loop((CLI While True Loop))
    Loop --> Chat
```

## 2. The `app.py` Streamlit Workflow

This is how the web UI manages the inference (chat) side. Notice how it acts as the orchestrator and talks directly to the component, using `@st.cache_resource` so the database does not reload on every single interaction.

```mermaid
graph TD
    Start(User opens localhost:8501) --> App[app.py Execution]
    App --> CacheFn{st.cache_resource: get_rag_chain}
    
    %% Setup (First Load Only)
    CacheFn -->|If not cached yet| CM([ConfigurationManager])
    CM -.-> YAML[(config.yaml & params.yaml)]
    CM -->|Returns Config Entity| RE[RAGEngine Component]
    RE -->|Loads| DB[(Saved artifacts/db)]
    RE -->|Builds| Chain[Setup Hybrid Retriever & ChatGroq LLM]
    Chain -->|Stores chain in memory| Cache[(Browser Cache)]
    
    %% Fast path (Subsequent loads)
    CacheFn -->|If already cached| Cache
    
    %% Execution
    Cache -.-> UI[Streamlit DOM Render]
    UI -->|User Types Question| Execute[rag_chain.invoke]
    Execute -->|Sends Answer| UI
```
