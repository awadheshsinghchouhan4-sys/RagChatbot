from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat
import streamlit as st

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("📄 PDF RAG Chatbot")

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# CHROMADB
# -----------------------------
@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

client = get_chroma_client()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

# -----------------------------
# PDF PROCESSING
# -----------------------------
if uploaded_file is not None:

    st.success("PDF Uploaded Successfully")

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # -----------------------------
    # CHUNKING
    # -----------------------------
    chunks = []

    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    st.write(f"Total Chunks: {len(chunks)}")

    # -----------------------------
    # COLLECTION
    # -----------------------------
    collection_name = "pdf_collection"

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    # -----------------------------
    # STORE EMBEDDINGS
    # -----------------------------
    with st.spinner("Generating Embeddings..."):

        for i, chunk in enumerate(chunks):

            embedding = embedding_model.encode(
                chunk
            ).tolist()

            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[chunk]
            )

    st.success("PDF Indexed Successfully")

    # -----------------------------
    # QUESTION INPUT
    # -----------------------------
    question = st.text_input(
        "Ask a Question"
    )

    if question:

        query_embedding = embedding_model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        context = "\n".join(
            results["documents"][0]
        )

        prompt = f"""
You are a helpful assistant.

Answer ONLY from the context.

Context:
{context}

Question:
{question}
"""

        with st.spinner("Generating Answer..."):

            response = chat(
                model="qwen2:1.5b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        answer = response["message"]["content"]

        st.subheader("Answer")
        st.write(answer)