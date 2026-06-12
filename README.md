---
title: PDFChat
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: false
---

# 📄 Chat with your PDF

A RAG-powered web application that lets you upload any PDF and ask 
questions about it in natural language.

## 🌐 Live Demo
**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/5hibuya/PDFChat)**

## 🧠 How it works

1. **PDF Loading** — document is read and text is extracted page by page
2. **Chunking** — text is split into overlapping chunks of ~1000 characters
3. **Embeddings** — each chunk is transformed into a 3072-dimension vector via Gemini Embeddings
4. **Indexing** — vectors are stored locally using FAISS
5. **Retrieval** — on each question, the 3 most relevant chunks are retrieved by similarity search
6. **Generation** — retrieved chunks are injected as context for Gemini to answer from

## 🛠️ Tech Stack
- Python 3.x
- Google Gemini API (google-genai) — LLM + Embeddings
- FAISS — local vector store
- PyPDF — PDF text extraction
- Streamlit — web interface
- python-dotenv — secure API key management

## 🚀 How to Run locally

```bash
pip install google-genai faiss-cpu pypdf python-dotenv streamlit
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```

## 📌 Notes
- Chunk overlap of 200 characters preserves context across boundaries
- Session state prevents reprocessing the PDF on every interaction
- The RAG approach surfaces only relevant sections per question,
  avoiding token limits and improving answer precision