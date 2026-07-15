from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from dotenv import load_dotenv
import os

load_dotenv()

chroma_db_directory=os.getenv('CHROMA_DB_DIRECTORY')

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), 
    persist_directory=chroma_db_directory
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

print("Top documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")
