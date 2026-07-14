from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), 
    persist_directory="/home/kubuntu/Documentos/curso_langchain/Tema 3/chroma_db"
)

retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k":2})

consulta = "¿En qual contrato es arrendadora Marta López Ruiz?"

resultados = retriever.invoke(consulta)

print("Top 2 resultados de la consulta:\n")

for i, doc in enumerate(resultados):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
