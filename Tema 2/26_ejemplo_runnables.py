from langchain_core.runnables import RunnableLambda

paso1 = RunnableLambda(lambda x: f"Numero 1: {x}")

def duplicar_texto(texto):
    return [texto]*2

paso2 = RunnableLambda(duplicar_texto)

cadena = paso1 | paso2
resultado = cadena.invoke(43)
print(resultado)