from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

plantilla_sistema = SystemMessagePromptTemplate.from_template(
    "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
)

plantilla_humana = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    plantilla_sistema,
    plantilla_humana
])

mensajes = chat_prompt.format_messages(
    rol="nutricionista",
    especialidad="dieta vegetariana",
    tono="profesional pero accesible",
    tema="proteinas vegetales",
    pregunta="¿Cuáles son las mejores fuentes de proteínas vegetales para alguien que sigue una dieta vegetariana?"
)

for msg in mensajes:
    print(msg.content)