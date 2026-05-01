import streamlit as st
import os
import shutil
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
DATA_DIR = ".data"
INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
CHAT_FILE = os.path.join(DATA_DIR, "chat_history.json")
MIN_CONFIDENCE = 35.0

# Page config
st.set_page_config(page_title="AI RAG App", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Styling elements for a modern clean UI */
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .uploaded-doc {
        border-left: 3px solid #00f2fe;
        padding-left: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(384) # 384 is dimension for all-MiniLM-L6-v2
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "auto_process" not in st.session_state:
    st.session_state.auto_process = True

def load_persistent_state():
    """Load FAISS index, metadata, and chat history from disk if present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(INDEX_FILE):
        try:
            st.session_state.faiss_index = faiss.read_index(INDEX_FILE)
        except Exception:
            st.session_state.faiss_index = faiss.IndexFlatL2(384)

    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                st.session_state.metadata = json.load(f)
        except Exception:
            st.session_state.metadata = []

    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                st.session_state.chat_history = json.load(f)
        except Exception:
            st.session_state.chat_history = []

def save_persistent_state():
    """Persist FAISS index, metadata, and chat history to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        faiss.write_index(st.session_state.faiss_index, INDEX_FILE)
    except Exception:
        pass

    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.metadata, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

load_persistent_state()

def remove_source_from_faiss(source_to_remove):
    """Remove all chunks associated with a specific source from FAISS and metadata in session state."""
    index = st.session_state.faiss_index
    metadata = st.session_state.metadata
    
    indices_to_keep = [i for i, meta in enumerate(metadata) if meta.get("source") != source_to_remove]
    
    if len(indices_to_keep) == 0:
        st.session_state.faiss_index = faiss.IndexFlatL2(384)
        st.session_state.metadata = []
    else:
        embeddings_to_keep = np.array([index.reconstruct(i) for i in indices_to_keep]).astype("float32")
        new_index = faiss.IndexFlatL2(384)
        new_index.add(embeddings_to_keep)
        st.session_state.faiss_index = new_index
        st.session_state.metadata = [metadata[i] for i in indices_to_keep]

    save_persistent_state()

def extract_text(file):
    """Extract text from PDF or TXT file."""
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file.name.endswith(".txt"):
        text = file.getvalue().decode("utf-8")
    return text

def chunk_text(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def embed_texts(texts, model):
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings).astype("float32")

def process_new_files(new_files):
    """Process uploaded files and add new chunks to the index."""
    if not new_files:
        st.warning("Please upload files first.")
        return

    with st.spinner("Processing documents..."):
        model = load_embedding_model()
        index = st.session_state.faiss_index
        metadata = st.session_state.metadata

        new_chunks = []
        new_metadata = []

        # Progress bar for files
        progress_bar = st.progress(0)
        for i, (file_id, file) in enumerate(new_files):
            text = extract_text(file)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            new_chunks.extend(chunks)
            # Use a unique identifier for the chunk based on existing metadata length
            start_idx = len(metadata) + len(new_metadata)
            new_metadata.extend([{"source": file.name, "chunk_id": start_idx + j, "text": chunk} for j, chunk in enumerate(chunks)])
            st.session_state.processed_files.add(file_id)
            progress_bar.progress((i + 1) / len(new_files))

        if new_chunks:
            st.info("Generating embeddings...")
            embeddings = embed_texts(new_chunks, model)
            index.add(embeddings)
            metadata.extend(new_metadata)
            st.success(f"Added {len(new_chunks)} chunks to the knowledge base!")
            save_persistent_state()
        else:
            st.warning("No readable text found in documents.")

@st.cache_resource
def load_embedding_model():
    """Load the SentenceTransformer model and cache it."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_groq_client():
    """Initialize Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
            
    if not api_key:
        st.error("GROQ_API_KEY not found. Please add it to your .env file locally, or to Streamlit Secrets when deploying.")
        st.stop()
    return Groq(api_key=api_key)
# Main UI setup
st.title("AI RAG Document Assistant")
st.markdown("Upload documents and ask questions based on their content.")

# Sidebar for controls
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True)

    st.session_state.auto_process = st.checkbox("Auto process uploads", value=st.session_state.auto_process)

    new_files = []
    if uploaded_files:
        for file in uploaded_files:
            size = getattr(file, "size", None)
            file_id = f"{file.name}:{size}" if size is not None else file.name
            if file_id not in st.session_state.processed_files:
                new_files.append((file_id, file))

    if st.session_state.auto_process:
        if new_files:
            process_new_files(new_files)
    else:
        if st.button("Process Documents"):
            process_new_files(new_files)
            
    st.divider()
    
    # Show stats
    index = st.session_state.faiss_index
    metadata = st.session_state.metadata
    st.metric("Total Chunks in DB", len(metadata))
    
    if metadata:
        st.write("**Chunk Distribution:**")
        source_counts = {}
        for meta in metadata:
            src = meta.get("source", "Unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            
        for source, count in source_counts.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"📄 {source}: {count} chunks")
            with col2:
                if st.button("X", key=f"del_{source}", help=f"Remove {source}"):
                    remove_source_from_faiss(source)
                    st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        save_persistent_state()
        st.rerun()
        
    if st.button("Clear Knowledge Base"):
        st.session_state.faiss_index = faiss.IndexFlatL2(384)
        st.session_state.metadata = []
        save_persistent_state()
        st.success("Knowledge base cleared!")
        st.rerun()

    if st.button("Reset Storage"):
        if os.path.exists(DATA_DIR):
            try:
                shutil.rmtree(DATA_DIR)
            except Exception:
                pass
        st.session_state.faiss_index = faiss.IndexFlatL2(384)
        st.session_state.metadata = []
        st.session_state.chat_history = []
        st.success("Storage reset. Upload documents again to rebuild the index.")
        st.rerun()

# Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("Show Retrieved Context"):
                for ctx in message["context"]:
                    st.markdown(f"**Source:** {ctx['source']}")
                    st.info(ctx['text'])
                    st.divider()

user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Display user query
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    save_persistent_state()
# Save call removed
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Retrieval
        model = load_embedding_model()
        index = st.session_state.faiss_index
        metadata = st.session_state.metadata
        
        retrieved_context = []
        context_text = ""
        out_of_scope_reason = None
        
        if index.ntotal > 0:
            query_embedding = embed_texts([user_query], model)
            distances, indices = index.search(query_embedding, k=3)

            # Cosine similarity on normalized vectors
            q = query_embedding[0].astype("float32")
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                q_norm = 1e-8
            q_unit = q / q_norm

            for idx in indices[0]:
                if idx != -1 and idx < len(metadata):
                    meta = metadata[idx].copy()
                    v = index.reconstruct(int(idx)).astype("float32")
                    v_norm = np.linalg.norm(v)
                    if v_norm == 0:
                        v_norm = 1e-8
                    v_unit = v / v_norm

                    cosine = float(np.dot(q_unit, v_unit))
                    score_pct = float((cosine + 1.0) / 2.0 * 100.0)
                    meta["confidence"] = score_pct

                    retrieved_context.append(meta)
                    context_text += (
                        f"\n--- Source: {meta['source']} (Match: {score_pct:.1f}%) ---\n"
                        f"{meta['text']}\n"
                    )
            if retrieved_context:
                max_conf = max([c.get("confidence", 0.0) for c in retrieved_context])
                if max_conf < MIN_CONFIDENCE:
                    out_of_scope_reason = "Low similarity to uploaded documents"
        else:
            context_text = "No documents available in the knowledge base."
            out_of_scope_reason = "No documents available"
            
        # Generation
        groq_client = get_groq_client()
        system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
    If the context doesn't contain the answer, say "I don't have enough information from the documents to answer that." 
    If the question is out of scope for the documents, clearly say you don't have enough document context and ask the user to upload relevant files.

FORMATTING INSTRUCTIONS:
- Keep your answer clear, concise, and well-formatted using Markdown.
- If the user asks for a quiz, list, or multiple-choice questions, ensure EACH option (e.g., A, B, C, D) starts on a NEW line.
- Use bold text for headings or emphasis.
- Use bullet points or numbered lists where appropriate.

CONTEXT:
{context_text}
"""
        try:
            if out_of_scope_reason:
                full_response = (
                    "I don't have enough information from the documents to answer that. "
                    "Please upload relevant files and try again."
                )
                message_placeholder.markdown(full_response)
            else:
                with st.spinner("Thinking..."):
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        model="llama-3.1-8b-instant", # Using Groq's fast llama3.1 model
                        temperature=0.5,
                        max_tokens=1024,
                    )
                
                full_response = chat_completion.choices[0].message.content
                message_placeholder.markdown(full_response)
            
            if retrieved_context:
                max_conf = max([c.get("confidence", 0.0) for c in retrieved_context])
                st.caption(f"Overall Confidence: {max_conf:.1f}%")
                with st.expander("Show Retrieved Context"):
                    for ctx in retrieved_context:
                        st.markdown(
                            f"**Source:** {ctx.get('source', 'Unknown')} "
                            f"(Match: {ctx.get('confidence', 0.0):.1f}%)"
                        )
                        st.info(ctx['text'])
                        st.divider()
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response,
                "context": retrieved_context
            })
            save_persistent_state()
        # Save call removed
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        # Save call removed
