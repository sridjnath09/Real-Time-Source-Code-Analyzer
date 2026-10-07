from src.helper import repo_ingestion,load_repo,text_splitter,load_embeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
load_dotenv()


os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embeddings(text_chunks)


vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory='./db'
)