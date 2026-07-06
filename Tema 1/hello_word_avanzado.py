from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda a {nombre} de manera cordial y amigable.",
)

chain = plantilla | llm
resultado = chain.invoke({"nombre": "Juan"})
print(resultado.content)