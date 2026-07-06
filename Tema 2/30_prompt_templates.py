from langchain_core.prompts import PromptTemplate

template = "Eres un experto en marketing. Sugiere un eslogan creativo para un producto {produto}."

prompt = PromptTemplate(
    template=template,
    input_variables=["produto"]
)

prompt_lleno = prompt.format(produto="zapatos deportivos")
print(prompt_lleno)