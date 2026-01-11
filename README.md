# PDF Question Answering with Grok + Streamlit + LangChain

A simple, clean **RAG** (Retrieval-Augmented Generation) chatbot that lets you ask questions about the content of a research paper (or any PDF) using **xAI's Grok** model.

<img width="876" height="622" alt="image" src="https://github.com/user-attachments/assets/7ef1612c-c905-48f2-901b-f2361d26db5e" />

## Features

- Upload once → chat forever with your PDF (currently fixed file path)
- Full document context is loaded and cached
- Clean, modern Streamlit interface
- Temperature control
- Chat history preserved during session
- Error handling for missing API key / file / model issues

## Tech Stack

- **Frontend** → Streamlit
- **LLM** → xAI Grok (via `langchain_xai`)
- **Document processing** → LangChain + PyPDFLoader
- **RAG** → Simple full-document context injection (no chunking/vectorstore yet)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pdf-qa-grok-streamlit.git
cd pdf-qa-grok-streamlit

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (or use Streamlit secrets)
echo "XAI_API_KEY=your_xai_api_key_here" > .env

# 5. Put your PDF file in the project root
#    Default filename: Research Paper - Final.pdf
#    (you can change the path in app.py)

# 6. Run the app
streamlit run app.py
