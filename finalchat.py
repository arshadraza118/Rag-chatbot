import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title("🚀 AI PDF Chatbot (RAG)")

# -------------------------
# API KEY
# -------------------------
# Priority: 1. Environment Variable (.env or OS) 2. Streamlit Secrets (Cloud)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        # st.secrets might raise an error if no secrets file exists locally
        api_key = None

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found! Add it in a .env file locally or in Streamlit Secrets on Cloud.")
    st.stop()

# -------------------------
# LOAD MODELS (CACHED)
# -------------------------
@st.cache_resource
def load_models():
    genai.configure(api_key=api_key)

    # ✅ Use standard model name without 'models/' prefix
    model = genai.GenerativeModel("gemini-1.5-flash")

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    return model, embed_model

model, embed_model = load_models()

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# -------------------------
# SIDEBAR (UPLOAD PDFs)
# -------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if files:
        text = ""

        for file in files:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""

        # SPLIT TEXT
        def split_text(text, size=500):
            if not text.strip():
                return []
            return [text[i:i+size] for i in range(0, len(text), size)]

        chunks = split_text(text)
        
        if not chunks:
            st.error("⚠️ No readable text found in the uploaded PDFs. Please ensure they are not scanned images.")
        else:
            st.session_state.chunks = chunks

            # EMBEDDINGS
            with st.spinner("Generating embeddings..."):
                embeddings = embed_model.encode(chunks)
                embeddings = np.array(embeddings).astype("float32")

                # FAISS INDEX
                if embeddings.shape[0] > 0:
                    dim = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(embeddings)
                    st.session_state.index = index
                    st.success(f"✅ Processed {len(chunks)} chunks successfully!")
                else:
                    st.error("⚠️ Failed to generate embeddings.")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# -------------------------
# SEARCH FUNCTION
# -------------------------
def search(query, k=5):
    if st.session_state.index is None or not st.session_state.chunks:
        return []

    q_vec = embed_model.encode([query]).astype("float32")
    D, I = st.session_state.index.search(q_vec, k)

    # Filter out -1 indices (FAISS default for no match) and ensure indices are valid
    valid_indices = [i for i in I[0] if 0 <= i < len(st.session_state.chunks)]
    return [st.session_state.chunks[i] for i in valid_indices]

# -------------------------
# DISPLAY CHAT
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
user_input = st.chat_input("Ask something about your PDFs...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    chunks = search(user_input)

    if not chunks:
        reply = "⚠️ Upload PDFs first."
    else:
        context = "\n".join(chunks)

        prompt = f"""
        You are a helpful AI assistant.

        Answer clearly using the context below.

        Context:
        {context}

        Question:
        {user_input}
        """

        # ✅ SAFE GENERATION
        try:
            response = model.generate_content(prompt)
            reply = response.text if hasattr(response, "text") else "⚠️ No response"
        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)