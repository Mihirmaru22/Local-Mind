# llm/generator.py
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from .prompt import RAG_PROMPT
from config.settings import MAIN_MODEL, OLLAMA_HOST

def get_rag_chain(model_name: str = None):
    target_model = model_name if model_name else MAIN_MODEL
    
    dynamic_llm = ChatOllama(
        model=target_model, 
        temperature=0, 
        num_ctx=32768, 
        keep_alive="24h", 
        base_url=OLLAMA_HOST,
        # THIS is where you disable thinking for maximum speed
        # options={"think": False}
        # THIS is the native LangChain flag to disable thinking completely 
          reasoning=False 
    )
    return RAG_PROMPT | dynamic_llm | StrOutputParser()