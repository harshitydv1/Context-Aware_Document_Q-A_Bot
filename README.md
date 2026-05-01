# Context-Aware_Document_Q&A_Bot: A Personal AI Document Assistant

Welcome to **Context-Aware_Document_Q&A_Bot**. This project lets you chat with your PDFs and text files through a simple Streamlit app. It uses Retrieval-Augmented Generation (RAG): your documents are indexed locally, and a fast LLM (Groq) answers questions using the most relevant chunks.

## How it works under the hood

When you upload a file, the app reads the text and splits it into smaller chunks (400 characters with a small overlap) so each chunk keeps enough context to be useful.

Next, it takes those chunks and converts them into numbers (embeddings) using a lightweight model running directly on your computer. These embeddings are stored in a local, file-based database called **FAISS**.

When you type a question, the app searches FAISS for the 3 most relevant chunks. It then sends those chunks and your question to **Llama 3** (via the Groq API). The response is grounded in the retrieved text.

## Tech Stack

* **Streamlit**: UI and app server.
* **FAISS**: Local vector search index.
* **Sentence Transformers**: `all-MiniLM-L6-v2` for embeddings.
* **Groq (Llama-3.1-8b)**: LLM for final answers.
* **Langchain**: `RecursiveCharacterTextSplitter` for chunking.
* **PyPDF2**: PDF text extraction.

## Features

- **Auto processing**: Uploading files triggers automatic indexing (no extra button).
- **Manual processing**: Turn off auto processing to use a "Process Documents" button.
- **Local indexing**: Your documents are embedded and indexed locally in FAISS.
- **Confidence score**: Each retrieved chunk shows a cosine-similarity match score, plus an overall confidence score.
- **Manage sources**: See per-file chunk counts and remove a file's chunks from the index.
- **Trust but verify**: View the exact chunks used to answer each question.
- **Reset storage**: Wipe local index and chat history with the Reset Storage button.

## Run it locally

If you want to spin this up locally, it's pretty straightforward.

Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Add your Groq API key in a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

Note: the FAISS index, metadata, and chat history are saved to a local .data folder and reloaded on startup. Delete the .data folder to fully reset, or use the Reset Storage button.
