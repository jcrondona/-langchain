from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

loader = PyPDFLoader("/home/kubuntu/Documentos/curso_langchain/Tema 3/Profile_juan_rondon.pdf")
docs = loader.load()

"""
for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("\n---\n")
"""

loader = WebBaseLoader("https://cavokaviacao.com/")
web = loader.load()

print(web)