import streamlit as st
import google.generativeai as genai 
import PyPDF2
import numpy as np
import faiss

# 🔑 Your Gemini API key
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("🚀 Advanced AI PDF Chatbot")

# -------------------------------
# SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_text" not in st.session_state:
    st.session_state.all_text = ""

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# -------------------------------
# SIDEBAR (UPLOAD + CONTROLS)
# -------------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        all_text = ""

        for file in uploaded_files:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text() + "\n"

        st.session_state.all_text = all_text

        # -----------------------
        # SPLIT TEXT
        # -----------------------
        def split_text(text, chunk_size=500):
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        chunks = split_text(all_text)
        st.session_state.chunks = chunks

        # -----------------------
        # CREATE VECTORS
        # -----------------------
        def text_to_vector(text):
            vector = np.zeros(300)
            for i, char in enumerate(text[:300]):
                vector[i] = ord(char)
            return vector.astype("float32")

        vectors = np.array([text_to_vector(c) for c in chunks])

        # -----------------------
        # BUILD FAISS INDEX
        # -----------------------
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        st.session_state.index = index

        st.success(f"{len(uploaded_files)} PDFs processed!")

    # Clear chat button
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# -------------------------------
# SEARCH FUNCTION
# -------------------------------
def search(query, k=5):
    index = st.session_state.index
    chunks = st.session_state.chunks

    if index is None:
        return []

    def text_to_vector(text):
        vector = np.zeros(300)
        for i, char in enumerate(text[:300]):
            vector[i] = ord(char)
        return vector.astype("float32")

    q_vec = text_to_vector(query).reshape(1, -1)
    D, I = index.search(q_vec, k)

    return [chunks[i] for i in I[0]]

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# USER INPUT
# -------------------------------
user_input = st.chat_input("Ask something about your PDFs...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------------------------
    # GET CONTEXT
    # ---------------------------
    relevant_chunks = search(user_input, k=5)

    if not relevant_chunks:
        ai_reply = "⚠️ Please upload PDFs first."
    else:
        context = "\n".join(relevant_chunks)

        prompt = f"""
        You are a helpful AI assistant.

        Give detailed and complete answers.

        Use this content:

        {context}

        Question: {user_input}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        ai_reply = response.text

    # Save AI reply
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    with st.chat_message("assistant"):
        st.markdown(ai_reply)