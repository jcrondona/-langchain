from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class AnalisisTexto(BaseModel):
    resumen:str = Field(description="Resumen breve del texto.")
    sentimiento:str = Field(description="Sentimiento general del texto, puede ser positivo, negativo o neutral.")
    
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)

structured_llm = llm.with_structured_output(AnalisisTexto)

texto_prueba = "No me gusto la pelicula mucho bla bla bla y poca acción."
resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

#print(resultado)
print(resultado.model_dump_json())
#print(type(resultado))
#print(dir(resultado))