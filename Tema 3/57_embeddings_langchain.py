from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
texto1 = "El aprendizaje automático es una rama de la inteligencia artificial que se centra en el desarrollo de algoritmos y modelos que permiten a las computadoras aprender de los datos y mejorar su rendimiento con el tiempo sin ser explícitamente programadas para cada tarea."
texto2 = "La inteligencia artificial es un campo de la informática que se enfoca en crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana, como el reconocimiento de voz, la visión por computadora y la toma de decisiones."

# texto1 = "La capital de venezuela es caracas"
# texto2 = "Caracas es la ciudad capital de venezuela"


vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)

print(f"Dimensión del vector 1: {len(vec1)}")
print(f"Dimensión del vector 2: {len(vec2)}")

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Similitud coseno entre los vectores v1 y v2: {cos_sim}")
