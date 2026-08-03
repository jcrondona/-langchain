from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

import streamlit as st
from config import *
from prompts import *

@st.cache_resource
def initialize_rag_system():
    # Vector store
    vectorestore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )
    
    #Modelos
    llm_query = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)
    
    # Retriever MMR (Maximal Marginal Relevance)
    base_retriever = vectorestore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )
    
    # Retriver adicional con similarity para comparar
    similarity_retriever = vectorestore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": SEARCH_K
        }
    )
    
    # Prompt personalizado para el MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)
    
    # MultiQueryRetriever con propmpt personalizado
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_query,
        prompt=multi_query_prompt
    )
    
    # Ensemble Retriever que combina MMR y Similarity
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[multi_query_retriever, similarity_retriever],
            weights=[0.7, 0.3],
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = multi_query_retriever    

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)
    
    # Funcion para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted = []
        
        for i,doc in enumerate(docs,1):
            header = f"[FRAGMENTO {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split('/')[-1] if '/' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                    
                if 'page' in doc.metadata:
                    header += f" - Página: {doc.metadata['page']}"
            
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted)
                    
    rag_chain = (
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )
    
    return rag_chain, multi_query_retriever

def query_rag(question):
    # try:
        rag_chain, retriever = initialize_rag_system()
        response = rag_chain.invoke(question)
        docs = retriever.invoke(question)
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else "") ,
                "fuente": doc.metadata.get('source', 'No especificada').split('/')[-1],
                "pagina": doc.metadata.get('page', 'No especificada'),
            }
            docs_info.append(doc_info)
        return response, docs_info
    # except Exception as e:
    #     print(f"Error al procesar la consulta: {str(e)}")
    #     error_message = f"Error al procesar la consulta: {str(e)}"
    #     return error_message, []
    
def get_retriever_info():
    
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": f"{SEARCH_K}",
        "diversidad": f"{MMR_DIVERSITY_LAMBDA}",
        "candidatos": f"{MMR_FETCH_K}",
        "umbral": f"{SIMILARITY_THRESHOLD}" if ENABLE_HYBRID_SEARCH else "N/A",
    }