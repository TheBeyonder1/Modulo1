import streamlit as st
import os
from rag import process_pdf, answer_question

st.set_page_config(page_title="Chatbot RAG PDF", layout="wide")

st.title("🤖 Chatbot RAG con PDF y Ollama")

# --- Sidebar para configuración del modelo ---
st.sidebar.header("Configuración del Modelo")
model_name = st.sidebar.selectbox(
    "Selecciona modelo",
    ["mistral:latest", "gemma:latest", "nomic-embed-text"]  # o la lista que quieras
)
temperature = st.sidebar.slider("Temperatura", 0.0, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)

uploaded_file = st.file_uploader("Sube tu PDF", type="pdf")
if uploaded_file:
    save_path = f"./data/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF cargado correctamente")

    if st.button("Procesar PDF"):
        process_pdf(save_path)
        st.success("PDF procesado y vectorizado")
        st.session_state["pdf_procesado"] = True

if "pdf_procesado" not in st.session_state:
    st.session_state["pdf_procesado"] = False

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state["pdf_procesado"]:
    user_question = st.text_input("Haz una pregunta sobre el PDF")
    if user_question:
        # Guardar la pregunta en el historial
        st.session_state.messages.append(("user", user_question))

        # Obtener respuesta con los parámetros de configuración
        response = answer_question(
            user_question,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        st.session_state.messages.append(("bot", response))

    # Mostrar historial de conversación
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"🧑‍💻 **Tú:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")
