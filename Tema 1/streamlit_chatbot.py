from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st


# Configurar la pagina de la app
st.set_page_config(page_title="Chatbox Basico", page_icon="🤖")
st.title("🤖 Chatbox Basico con Langchain")
st.markdown("Este es un ejemplo de un chatbot básico utilizando Langchain y Streamlit. Escribe un mensaje y el chatbot te responderá.")

with st.sidebar:
    st.header("Configuración del Chatbot")
    temperature = st.slider("Temperatura del modelo", min_value=0.0, max_value=1.0, value=0.5)
    model_name = st.selectbox("Selecciona el modelo", options=["gemini-2.5-flash"])
    button_limpiar = st.button("Limpiar Conversación")

chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

#Inicializar el historial de mensajes
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
    
# Definir la plantilla de prompt para el chatbot
prompt_template = PromptTemplate(
    input_variables=["pregunta", "historial"],
    template="""Eres un asistente útil y amigable llamado Chatbot. 
    Historial de conversación:
    {historial}
    Responde de manera clara y concisa a la siguiente pregunta:: {pregunta}"""
)

# Crear cadena usando LCEL (Langchain Expression Language)
chain = prompt_template | chat_model

# Mostar mensajes anteriores en la interfaz
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        # No mostrar mensajes del sistema
        continue
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    
    with st.chat_message(role):
        st.markdown(msg.content)
        
if button_limpiar:
    st.session_state.mensajes = []
    st.rerun()
    
# Cuando de entrada de texto del usuario
pregunta = st.chat_input("Escribe tu mensaje aquí...")
if pregunta:
    # Mostar inmediatamente el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(pregunta)
        
    try:
        
        # Mostrar la respuesta del modelo en la interfaz 
        with st.chat_message("assistant"):
            response_placeholder = st.empty()  # Placeholder para mostrar la respuesta a medida que llega
            full_response = ""
            
            # Obtener la respuesta del modelo en streaming
            for chunk in chain.stream({"pregunta": pregunta, "historial": st.session_state.mensajes}):
                full_response += chunk.content
                response_placeholder.markdown(full_response + " |")  # Actualizar el placeholder con la respuesta parcial

            response_placeholder.markdown(full_response)  # Mostrar la respuesta completa al finalizar el streaming
        
            # Almacenamos el mensaje en la memoria de streamlit
            st.session_state.mensajes.append(HumanMessage(content=pregunta))
            # Almacenamos la respuesta del modelo en la memoria de streamlit
            st.session_state.mensajes.append(AIMessage(content=full_response))
            
    except Exception as e:
        st.error(f"Error al obtener respuesta del modelo: {str(e)}")
        st.info("Verifica que tu API key sea válida.")   
    

    
    


        
