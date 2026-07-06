from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso."),
    ("human", "{text}"),
])

mensajes = chat_prompt.format_messages(text="Hola mundo, ¿Cómo estás?")

for mensaje in mensajes:
    print(f"{type(mensaje)}: {mensaje.content}")