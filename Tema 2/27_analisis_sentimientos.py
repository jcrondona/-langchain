import json

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def preprocess_text(text: str) -> str:
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    text = text.strip()
    text = text[:500]
    return text

processor = RunnableLambda(preprocess_text)

def generate_summary(text: str) -> str:
    """Genera un resumen conciso del texto"""
    prompt = f"Resumen en una sola oración de forma clara y concisa: {text}"
    response = llm.invoke(prompt)
    return response.content

summary_branch = RunnableLambda(generate_summary)

def analyse_sentiment(text: str) -> str:
    """Analiza el sentimiento y devuelve resultado estructurado"""
    
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde SOLAMENTE con un JSON con el siguiente formato:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    Responde EXCLUSIVAMENTE con un objeto JSON válido.
    NO uses markdown.
    NO uses ```json.
    NO añadas explicaciones.
    
    Texto: {text}"""
     
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}

sentiment_branch = RunnableLambda(analyse_sentiment)

def merge_results(data: dict) -> dict:
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }
    
merger = RunnableLambda(merge_results)

parraller_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch
})

chain = processor | parraller_analysis | merger

reviews_batch = [
    "Excelente producto, muy satisfecho con la compra",
    "Terrible calidad, no lo recomiendo para nada",
    "Está bien, cumple su función básica pero nada especial"
]

resultado_batch = chain.batch(reviews_batch)

print(resultado_batch)