# Cuestionario Modulo 1
Este proyecto forma parte de la especialización en Inteligencia Artificial. Consiste en desarrollar un chatbot interactivo capaz de responder preguntas sobre el contenido de un documento PDF, utilizando la técnica RAG (Retrieval-Augmented Generation).

## ⚙️ Instalación y Configuración

### 1. Clona el repositorio

```bash
git clone https://github.com/TheBeyonder1/Modulo1
```

### 2. Clona el repositorio Crea y activa un entorno virtual
```bash
uv venv
uv pip install -r requirements.txt
```
### 3. Asegurate de corrrer ollama 
```bash
ollama list
ollama serve
```
### 4. Configura las variables de entorno
Crea un archivo en la raiz del proyecto con la estructura del env.example con la apikey de tu langsmith

### 5 Corre el proyecto
```bash
uv streamlit run src/app.py
```


Aqui hay algunas imagenes del proceso y de la interfaz
![alt text](/images/image2.png)

Evidencia de Langsmith
![alt text](/images/image.png)

