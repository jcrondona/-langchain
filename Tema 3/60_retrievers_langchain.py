from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

chroma_db_directory=os.getenv('CHROMA_DB_DIRECTORY')

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), 
    persist_directory=chroma_db_directory
)

retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k":2})

consulta = "¿En qual contrato es arrendadora Marta López Ruiz?"

resultados = retriever.invoke(consulta)

print("Top 2 resultados de la consulta:\n")

for i, doc in enumerate(resultados):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
