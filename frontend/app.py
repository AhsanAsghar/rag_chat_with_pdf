"""
Simple chat UI for the RAG backend. Lets the user upload a PDF/notes
file, then ask questions in a chat interface, with answers grounded
in the uploaded document(s) and source citations shown underneath.

Run with: streamlit run frontend/app.py
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Chat with your PDF/Notes", page_icon="📚", layout="wide")

# ---------- Session state ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content}
if "documents" not in st.session_state:
    st.session_state.documents = []


def refresh_documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        resp.raise_for_status()
        st.session_state.documents = resp.json().get("documents", [])
    except requests.RequestException as e:
        st.sidebar.error(f"Could not reach backend: {e}")


# ---------- Sidebar: upload + document list ----------
with st.sidebar:
    st.title("📚 Chat with your Docs")
    st.caption("Upload a PDF or notes file, then ask questions about it.")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "md", "txt"],
        help="PDF, Markdown, or plain text files are supported.",
    )

    if uploaded_file is not None:
        if st.button("Ingest document", type="primary", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'... this can take a moment."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(data["message"])
                        refresh_documents()
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")

    st.divider()
    st.subheader("Uploaded documents")
    refresh_documents()

    if not st.session_state.documents:
        st.info("No documents uploaded yet.")
    else:
        for doc in st.session_state.documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 **{doc['filename']}**")
                st.caption(f"{doc['num_chunks']} chunks")
            with col2:
                if st.button("🗑️", key=f"del_{doc['document_id']}"):
                    requests.delete(f"{BACKEND_URL}/documents/{doc['document_id']}")
                    refresh_documents()
                    st.rerun()

    st.divider()
    if st.button("Clear all documents + chat", use_container_width=True):
        requests.post(f"{BACKEND_URL}/reset")
        st.session_state.chat_history = []
        refresh_documents()
        st.rerun()

# ---------- Main: chat interface ----------
st.header("Ask questions about your documents")

if not st.session_state.documents:
    st.warning("Upload a document from the sidebar to get started.")

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    page_info = f" (page {src['page']})" if src.get("page") else ""
                    st.markdown(
                        f"**{src['filename']}{page_info}** "
                        f"— similarity: {src['similarity_score']:.2f}"
                    )
                    st.caption(src["text_snippet"])

# Chat input
question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send prior turns (excluding sources) as history for
                # follow-up-question context.
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                    if m["role"] in ("user", "assistant")
                ]

                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question, "history": history_payload},
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                page_info = f" (page {src['page']})" if src.get("page") else ""
                                st.markdown(
                                    f"**{src['filename']}{page_info}** "
                                    f"— similarity: {src['similarity_score']:.2f}"
                                )
                                st.caption(src["text_snippet"])

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    error_msg = f"Error: {resp.json().get('detail', resp.text)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )

            except requests.RequestException as e:
                error_msg = f"Could not reach backend: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )
