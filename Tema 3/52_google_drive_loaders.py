from langchain_google_community import GoogleDriveLoader

credentials_path = "/home/kubuntu/Documentos/curso_langchain/Tema 3/credentials.json"
token_path = "/home/kubuntu/Documentos/curso_langchain/Tema 3/token.json"
loader = GoogleDriveLoader(
    folder_id="1b2B1Sh7Reic5Uj0eUBV1EdeU3b6e90RR",
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True
)

documents = loader.load()

for document in documents:
    print(document.metadata["title"])