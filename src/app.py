import streamlit as st
import os
from rag import process_pdf

st.set_page_config(page_title="Chatbot RAG PDF", layout="wide")

st.title("🤖 Chatbot RAG con PDF y Ollama")

# Subida del PDF
uploaded_file = st.file_uploader("📄 Sube un documento PDF", type="pdf")

if uploaded_file is not None:
    # Guardar el PDF temporalmente en la carpeta data/
    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Archivo {uploaded_file.name} guardado exitosamente.")

    if st.button("🔍 Procesar PDF"):
        with st.spinner("Procesando documento..."):
            process_pdf(save_path)
        st.success("📚 ¡PDF procesado y almacenado en ChromaDB!")

