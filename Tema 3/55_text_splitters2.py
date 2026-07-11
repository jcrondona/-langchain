from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

#1 Cargar el documento PDF
loader = PyPDFLoader("/home/kubuntu/Documentos/curso_langchain/Tema 3/quijote.pdf")
pages = loader.load()

#2 Dividir el texto en partes más pequeñas utilizando RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, 
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

#3 Pasar cada fragmento al modelo de lenguaje para generar un resumen
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
summaries = []
for chunk in chunks:
    response = llm.invoke(f"Has un resumen de los puntos mas importantes del siguiente fragmento: {chunk.page_content}")
    summaries.append(response)

#4 Combinar los resúmenes de cada fragmento en un resumen final
final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente, conciso y completo: {' '.join(summaries)}")
print(final_summary.content)