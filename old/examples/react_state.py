from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Any, Optional, TypedDict, List
import os
from dotenv import load_dotenv
import warnings
from IPython.display import Image, display



warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.1,
    convert_system_message_to_human=True,
    transport="rest"
)

@tool
def get_weather(city: str) -> str:
    """Obtiene el clima para una ciudad dada"""
    return f"Bad weather all the day {city}"

@tool
def get_motivation(question: str) -> str:
    """Obtiene un mensaje motivacional para una pregunta dada"""
    return f"Everything is going to be fine about '{question}'"

tools = [get_weather, get_motivation]

system_prompt = """
You are a helpful assistant that reasons step by step before answering.
Always check your tools before answering.
"""

class ToolResponse(BaseModel):
    tool: str
    output: Any
    message: Optional[str] = None

from langgraph.prebuilt import create_react_agent

react_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
    response_format=ToolResponse,
)

# Estado compartido
class AgentState(TypedDict):
    messages: List[Any]
    structured_response: Optional[ToolResponse]
    summary: Optional[str]

# Nodo para resumir los mensajes previos
def summarize_step(state: AgentState) -> AgentState:
    conversation = "\n".join([msg.content for msg in state["messages"]])
    prompt = f"Resume brevemente la conversación para ayudar en la respuesta:\n{conversation}"

    summary_result = llm.invoke([HumanMessage(content=prompt)])
    summary_text = summary_result.content  # <-- Aquí el cambio

    new_state = state.copy()
    new_state["summary"] = summary_text
    return new_state

# Nodo para ejecutar el agente con el resumen
def agent_step(state: AgentState) -> AgentState:
    # Incluir el resumen en el input para el agente (por ejemplo en un mensaje adicional)
    messages = state["messages"][:]
    if state.get("summary"):
        messages.append(HumanMessage(content=f"Resumen previo: {state['summary']}"))

    result = react_agent.invoke({"messages": messages})
    return {
        "messages": result["messages"],
        "structured_response": result.get("structured_response"),
        "summary": state.get("summary")
    }

# Crear y compilar grafo con 2 nodos
workflow = StateGraph(AgentState)
workflow.add_node("summarize", summarize_step)
workflow.add_node("agent_step", agent_step)
workflow.set_entry_point("summarize")
workflow.add_edge("summarize", "agent_step")
workflow.set_finish_point("agent_step")
graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

png_bytes = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)



def main():
    user_input = input("Escribe tu mensaje para el agente: ")
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "structured_response": None,
        "summary": None,
    }
    final_state = graph.invoke(initial_state)

    print("\n--- Resumen ---")
    print(final_state.get("summary"))
    print("\n--- Respuesta estructurada ---")
    print(final_state["structured_response"])

    """ print("\n--- Historial de mensajes ---")
    for msg in final_state["messages"]:
    print(f"{msg.__class__.__name__}:\n{msg.content}\n{'-'*40}") """

if __name__ == "__main__":
    main()
    
