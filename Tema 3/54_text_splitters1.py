from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

#1 Cargar el documento PDF
loader = PyPDFLoader("/home/kubuntu/Documentos/curso_langchain/Tema 3/quijote.pdf")
pages = loader.load()

#2 Combinar todas las páginas en un solo texto
full_text = ""
for page in pages:
    full_text += page.page_content + "\n"
    
#3 Pasar el texto completo al modelo de lenguaje para generar un resumen
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
#Erro de memoria, el texto es demasiado largo para procesarlo de una sola vez. Se recomienda dividir el texto en partes más pequeñas y procesarlas por separado.
#Não executar va consumir os tokens de forma muito rápida e pode gerar erros de memória.
response = llm.invoke(f"Has un resumen de los puntos mas importantes del siguiente documento: {full_text}")

print(response)