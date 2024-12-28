# FINDER

A tool to help you search through research papers and other documents quickly and effectively. This tool uses Adaptive RAG to generate answers based on information from your uploaded documents. If no relevant information is found in your documents, FINDER scours the internet to find relevant information before replying.

> **Adaptive RAG** is a strategy for Retrieval-Augmented Generation (RAG) that unites (1) query analysis with (2) active/self-corrective RAG.

![Adaptive RAG Architecture](/static/rag_archi.png)  
[Read more about self-corrective RAG](https://blog.langchain.dev/agentic-rag-with-langgraph/)

---

## Features

- Efficient parsing and hierarchical context retention for complex documents.  
- Semantic chunking with fallback mechanisms for optimal input size.  
- Embedding and retrieval using state-of-the-art methods for accuracy.  
- Internet augmentation for queries when local information is insufficient.  
- Intuitive web app with two-way communication using WebSockets.  

---

## System Architecture

### 1. Indexing

- **Parsing**:  
  Marker, an open-source parser, is used to parse PDFs into markdown and chunk documents by headers.  

- **Context Retention**:  
  Parsed markdown is converted into an Abstract Syntax Tree (AST). A Breadth-First Search (BFS) is applied to prepend header paths to the metadata of each chunk, retaining hierarchical context.  

- **Chunking**:  
  - Context windows that are too large reduce accuracy, causing the model to ignore explicit prompts.  
  - Semantically chunk documents into smaller parts by identifying gradients in semantic changes. This is particularly effective for research papers with high inter-chunk correlation.  
  - Remaining large chunks are split using LangChain's RecursiveCharacterTextSplitter, which respects sentence, paragraph, and document structure while adhering to token limits.  
  [Learn more about semantic chunking](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)  
  [LangChain RecursiveCharacterTextSplitter documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)  

- **Embedding**:  
  A lightweight model was selected from the MTEB leaderboard to embed chunks and queries efficiently.  

- **Database**:  
  Milvus, an open-source vector database, is used for storage. It provides high efficiency across various environments.  
  [Milvus documentation](https://milvus.io/docs/overview.md)  

### 2. Retrieval

- Used merged-rank retrieval to search and retrieve information from relevant documents.  
- Retrieved and ranked documents by cosine similarity.  

### 3. Generation
- Follows the adaptive rag workflow to generate a reply. The workflow is created using LangGraph.
- Utilized the Llama 3 model from Ollama for grading and generating responses.  

### 4. Web Application

- Built with Flask and Jinja for a user-friendly interface.  
- Implemented two-way communication using WebSockets.  

---

## How to Run

1. Install all dependencies as specified in `requirements.txt`.  
2. Setup Milvus Container, follow instruction [here](https://milvus.io/docs/install_standalone-docker.md)
3. Run the application:  
   ```bash
   python app.py