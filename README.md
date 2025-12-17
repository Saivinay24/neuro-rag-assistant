#  Neuro-RAG-Assistant

> **A local-first Retrieval-Augmented Generation (RAG) system for querying dense neuroscience literature.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 📌 Overview
**Neuro-RAG-Assistant** is a specialized semantic search tool designed to help students and researchers navigate complex neuroscience textbooks. Unlike standard keyword search, this system uses **vector embeddings** to understand the *context* of a query (e.g., "How do glial cells support neurons?") and retrieves precise answers from a local knowledge base.

It runs entirely offline using a quantized **Phi-3** model, ensuring data privacy and zero inference costs.

##  Key Features
* **Local Inference:** Powered by `Phi-3-mini` (via `llama.cpp`) for high performance on consumer hardware (CPU/Mac Metal).
* **Semantic Retrieval:** Uses FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`) to find relevant text chunks.
* **Citation Tracking:** Returns the specific source chunks used to generate the answer, promoting scientific reproducibility.
* **Privacy Focused:** No data is sent to external APIs (OpenAI/Anthropic).

##  Tech Stack
* **LLM:** Microsoft Phi-3 Mini (4k context, Quantized GGUF)
* **Orchestration:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Interface:** Streamlit
* **Ingestion:** PyPDF

---

## ⚡ Quick Setup

To get the system running on your local machine, run the following commands in your terminal:

```bash
# Clone the repository
git clone [https://github.com/Saivinay24/neuro-rag-assistant.git](https://github.com/Saivinay24/neuro-rag-assistant.git)

# Navigate to the project directory
cd neuro-rag-assistant

# Create and activate a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
# Data Ingestion
 Before running the chat interface, you must build the "brain" (Vector Index) by feeding it a textbook or paper.
 Place your neuroscience PDF file (e.g., textbook.pdf) inside the data/ folder.
 Run the ingestion script to process the text and create the FAISS index: 
```bash
python -m app.ingest
```
You will see a "Ingestion complete" message and a new faiss_index folder will appear.
# Launch the Assistant
 Start the web interface:
```bash
streamlit run app/ui.py
```

