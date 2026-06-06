import streamlit as st
import numpy as np
import faiss
import io
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ============================================================
# RAG FUNCTIONS
# ============================================================

def split_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)

def build_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def ask(question, chunks, index, top_k=3):
    q_emb = get_embedding(question).reshape(1, -1)
    _, indices = index.search(q_emb, top_k)
    context = "\n\n".join([chunks[i] for i in indices[0]])
    prompt = f"""Use the context below to answer the question.
    
    Context:
    {context}

    Question: {question}"""
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text

# ============================================================
# UI
# ============================================================

st.title("📄 Chat with your PDF")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    if "chunks" not in st.session_state: # Se as chunks não existirem
        with st.spinner("Processing PDF..."): # Sinalize ao usuário o processamento
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join([p.extract_text() for p in reader.pages]) # O que é esse extract_text
            st.session_state.chunks = split_chunks(text) # Session state para não perder o chunks no recarregamento da página, vira variável
            st.session_state.index = build_index(st.session_state.chunks)
        st.success(f"Ready! {len(st.session_state.chunks)} chunks indexed.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        st.chat_message(message["role"]).write(message["content"]) # Não entendi a função desse for, essa é a parte que reescreve toda a conversa no recarregamento da tela?

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.chat_message("user").write(question)
        st.session_state.history.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            answer = ask(question, st.session_state.chunks, st.session_state.index)

        st.chat_message("assistant").write(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})