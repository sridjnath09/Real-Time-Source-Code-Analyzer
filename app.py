from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from git import Repo
from langchain_chroma import Chroma
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "repo"
DB_DIR = BASE_DIR / "db"
DEFAULT_REPO_URL = "https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"

def ensure_env():
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
You are a software engineering assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context,
say that you don't have enough information.

Context:
{context}

Question:
{question}
"""
    )

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def clone_repo(repo_url: str = DEFAULT_REPO_URL) -> Path:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    Repo.clone_from(repo_url, to_path=str(REPO_DIR))
    return REPO_DIR

def build_vectorstore(force_rebuild: bool = False):
    embeddings = get_embeddings()

    if force_rebuild and DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    if DB_DIR.exists() and not force_rebuild:
        try:
            return Chroma(
                persist_directory=str(DB_DIR),
                embedding_function=embeddings,
            )
        except Exception:
            pass

    repo_path = REPO_DIR if REPO_DIR.exists() else clone_repo()

    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*.py",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON),
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=500,
        chunk_overlap=50,
        language=Language.PYTHON,
    )

    chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )

    return vectordb

def get_llm():
    ensure_env()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or your environment.")
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
    )

def normalize_response(result):
    if hasattr(result, "content"):
        return str(result.content)

    if isinstance(result, dict):
        for key in ("answer", "response", "text", "output_text"):
            if key in result:
                return str(result[key])

    if isinstance(result, list):
        return "\n".join(normalize_response(item) for item in result)

    return str(result)

def get_rag_chain():
    vectordb = build_vectorstore(force_rebuild=False)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    prompt = build_prompt()
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ingest", methods=["POST"])
def ingest_repo():
    try:
        repo_url = request.form.get("repo_url") or DEFAULT_REPO_URL
        clone_repo(repo_url)
        build_vectorstore(force_rebuild=True)
        return jsonify({"status": "success", "message": "Repository ingested and indexed."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        question = (request.form.get("question") or request.form.get("msg") or "").strip()
        if not question:
            return jsonify({"response": "Please enter a question."}), 400

        rag_chain = get_rag_chain()
        result = rag_chain.invoke(question)
        return jsonify({"response": normalize_response(result)})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/get", methods=["GET", "POST"])
def get_answer():
    try:
        question = request.args.get("question") or request.form.get("question") or request.form.get("msg") or ""
        if not question:
            return jsonify({"response": "Please enter a question."}), 400

        rag_chain = get_rag_chain()
        result = rag_chain.invoke(question)
        return jsonify({"response": normalize_response(result)})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=    from __future__ import annotations
    
    import os
    import shutil
    from pathlib import Path
    
    from dotenv import load_dotenv
    from flask import Flask, jsonify, render_template, request
    from git import Repo
    from langchain_chroma import Chroma
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import LanguageParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
    
    load_dotenv()
    
    app = Flask(__name__)
    
    BASE_DIR = Path(__file__).resolve().parent
    REPO_DIR = BASE_DIR / "repo"
    DB_DIR = BASE_DIR / "db"
    DEFAULT_REPO_URL = "https://github.com/entbappy/End-to-end-Medical-Chatbot-Generative-AI"
    
    def ensure_env():
        if not os.getenv("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""
    
    def build_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """
    You are a software engineering assistant.
    
    Answer the question using only the provided context.
    
    If the answer cannot be found in the context,
    say that you don't have enough information.
    
    Context:
    {context}
    
    Question:
    {question}
    """
        )
    
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def clone_repo(repo_url: str = DEFAULT_REPO_URL) -> Path:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
    
        Repo.clone_from(repo_url, to_path=str(REPO_DIR))
        return REPO_DIR
    
    def build_vectorstore(force_rebuild: bool = False):
        embeddings = get_embeddings()
    
        if force_rebuild and DB_DIR.exists():
            shutil.rmtree(DB_DIR)
    
        if DB_DIR.exists() and not force_rebuild:
            try:
                return Chroma(
                    persist_directory=str(DB_DIR),
                    embedding_function=embeddings,
                )
            except Exception:
                pass
    
        repo_path = REPO_DIR if REPO_DIR.exists() else clone_repo()
    
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*.py",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON),
        )
    
        documents = loader.load()
    
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=500,
            chunk_overlap=50,
            language=Language.PYTHON,
        )
    
        chunks = text_splitter.split_documents(documents)
    
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
        )
    
        return vectordb
    
    def get_llm():
        ensure_env()
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or your environment.")
        return ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0,
        )
    
    def normalize_response(result):
        if hasattr(result, "content"):
            return str(result.content)
    
        if isinstance(result, dict):
            for key in ("answer", "response", "text", "output_text"):
                if key in result:
                    return str(result[key])
    
        if isinstance(result, list):
            return "\n".join(normalize_response(item) for item in result)
    
        return str(result)
    
    def get_rag_chain():
        vectordb = build_vectorstore(force_rebuild=False)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        prompt = build_prompt()
        llm = get_llm()
    
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
    
        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
    
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")
    
    @app.route("/ingest", methods=["POST"])
    def ingest_repo():
        try:
            repo_url = request.form.get("repo_url") or DEFAULT_REPO_URL
            clone_repo(repo_url)
            build_vectorstore(force_rebuild=True)
            return jsonify({"status": "success", "message": "Repository ingested and indexed."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/chatbot", methods=["POST"])
    def chatbot():
        try:
            question = (request.form.get("question") or request.form.get("msg") or "").strip()
            if not question:
                return jsonify({"response": "Please enter a question."}), 400
    
            rag_chain = get_rag_chain()
            result = rag_chain.invoke(question)
            return jsonify({"response": normalize_response(result)})
        except Exception as e:
            return jsonify({"response": f"Error: {str(e)}"}), 500
    
    @app.route("/get", methods=["GET", "POST"])
    def get_answer():
        try:
            question = request.args.get("question") or request.form.get("question") or request.form.get("msg") or ""
            if not question:
                return jsonify({"response": "Please enter a question."}), 400
    
            rag_chain = get_rag_chain()
            result = rag_chain.invoke(question)
            return jsonify({"response": normalize_response(result)})
        except Exception as e:
            return jsonify({"response": f"Error: {str(e)}"}), 500
    
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8080, debug=True)8080, debug=True)