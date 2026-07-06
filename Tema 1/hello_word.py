from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

pregunta = "¿Cuál es la capital de Francia?"
print("Pregunta:", pregunta)
respuesta = llm.invoke(pregunta)
print("Respuesta:", respuesta.content)