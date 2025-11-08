import chromadb
from langchain_community.document_loaders import pyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Setting environments
DATA_PATH="./data"
CHROMA_PATH="./chroma_db"
chroma_client=chromadb.PersistentClient(path=CHROMA_PATH)
collection=chroma_client.get_or_create_collection(name="documents")

#Loading PDF files from data folder
loader =pyPDFDirectoryLoader(DATA_PATH)

raw_doc=loader.load()

#Splitting doc

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunks=text_splitter.split_documents(raw_doc)

#Adding documents to ChromaDB
collection.upsert(documents=[doc.page_content for doc in chunks],
                  metadatas=[doc.metadata for doc in chunks])
print(f"Added {len(chunks)} documents to ChromaDB collection 'documents'.")
