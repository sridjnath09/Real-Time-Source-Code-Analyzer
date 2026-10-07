
from dotenv import load_dotenv
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)
from langchain_community.vectorstores import FAISS
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
from git import Repo


#Clone the repo
def repo_ingestion(repo_url):
    os.makedirs("repo", exist_ok=True)
    repo_path="repo/"
    Repo.clone_from(repo_url, to_path=repo_path)

 #loading repo as documents
def load_repo(repo_path):
    loader=GenericLoader.from_filesystem(repo_path,
                                      glob="**/*",
                                      suffixes=[".py"],
                                      parser=LanguageParser(language=Language.PYTHON)
                                      )
    documents = loader.load()
    return documents

#Creating text chunks
def text_splitter(documents):
    documents_splitter=RecursiveCharacterTextSplitter.from_language(
    chunk_size=500,
    chunk_overlap=20,
    language=Language.PYTHON)

    text_chunks=documents_splitter.split_documents(documents)
    return text_chunks


#load embeddings
def load_embeddings(text_chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    return embeddings