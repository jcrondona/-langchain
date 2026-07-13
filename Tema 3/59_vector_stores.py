from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

loader = PyPDFDirectoryLoader("/home/kubuntu/Documentos/curso_langchain/Tema 3/contratos")
documentos = loader.load()

print(f"Documentos cargados: {len(documentos)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

docs_split = text_splitter.split_documents(documentos)

print(f"Documentos divididos: {len(docs_split)}")

vectorstore = Chroma.from_documents(
    docs_split, 
    OpenAIEmbeddings(model="text-embedding-3-large"), 
    persist_directory="/home/kubuntu/Documentos/curso_langchain/Tema 3/chroma_db"
)

consulta = "¿En qual contrato es arrendadora Marta López Ruiz?"

resultados = vectorstore.similarity_search(consulta, k=3)

print("Top 3 resultados de la consulta:\n")

for i, doc in enumerate(resultados):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
