import os
from dotenv import load_dotenv
import glob
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)
embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model="llama3.2:1b"

KNOWLEDGE_BASE=str(Path(__file__).parent/"knowledge_base")
DB_NAME=str(Path(__file__).parent/"vector_db")

def fetch_documents():

    loader = DirectoryLoader(
        KNOWLEDGE_BASE,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    for doc in documents:
        doc.metadata["doc_type"] = "pdf"

    return documents

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
        
    

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embedding_model).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
