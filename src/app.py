import streamlit as st
import os
from rag import process_pdf,answer_question

st.set_page_config(page_title="Chatbot RAG PDF", layout="wide")

st.title("🤖 Chatbot RAG con PDF y Ollama")


uploaded_file = st.file_uploader("Sube tu PDF", type="pdf")
if uploaded_file:
    save_path = f"./data/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF cargado correctamente")

    if st.button("Procesar PDF"):
        process_pdf(save_path)
        st.success("PDF procesado y vectorizado")

if "pdf_procesado" not in st.session_state:
    st.session_state["pdf_procesado"] = False

if st.button("Iniciar chat"):
    st.session_state["pdf_procesado"] = True

if st.session_state["pdf_procesado"]:
    user_question = st.text_input("Haz una pregunta sobre el PDF")
    if user_question:
        response = answer_question(user_question)
        st.markdown("**Respuesta:**")
        st.write(response)
