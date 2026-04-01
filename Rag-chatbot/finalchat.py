import streamlit as st
import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title("🚀 AI PDF Chatbot (RAG)")

# -------------------------
# LOAD API KEY (STREAMLIT CLOUD)
# -------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# LOAD EMBEDDING MODEL
# -------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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
# SIDEBAR (UPLOAD)
# -------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    files = st.file_uploader("Upload", type="pdf", accept_multiple_files=True)

    if files:
        text = ""

        for file in files:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        # SPLIT TEXT
        def split_text(text, size=500):
            return [text[i:i+size] for i in range(0, len(text), size)]

        chunks = split_text(text)
        st.session_state.chunks = chunks

        # CREATE EMBEDDINGS
        embeddings = embed_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        # FAISS INDEX
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        st.session_state.index = index

        st.success(f"{len(files)} PDFs processed!")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# -------------------------
# SEARCH FUNCTION
# -------------------------
def search(query, k=5):
    if st.session_state.index is None:
        return []

    q_vec = embed_model.encode([query]).astype("float32")

    D, I = st.session_state.index.search(q_vec, k)

    return [st.session_state.chunks[i] for i in I[0]]

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

    # GET CONTEXT
    relevant_chunks = search(user_input)

    if not relevant_chunks:
        reply = "⚠️ Please upload PDFs first."
    else:
        context = "\n".join(relevant_chunks)

        prompt = f"""
        You are a helpful AI assistant.

        Answer clearly and in detail using the context below.

        Context:
        {context}

        Question:
        {user_input}
        """

        response = model.generate_content(prompt)
        reply = response.text

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
