from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Any, Optional
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()  # Esto debe estar ANTES de os.getenv()

class WeatherResponse(BaseModel):
    conditions: str
    description: str


class ToolResponse(BaseModel):
    tool: str                    # Nombre de la herramienta usada
    output: Any                  # Resultado crudo que devuelva la herramienta
    message: Optional[str] = None  # Respuesta final (si quieres incluirla)

# Asegúrate de que la API key esté configurada
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY no está configurada en las variables de entorno")

# Define el modelo con la configuración correcta
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Modelo disponible
    google_api_key=api_key,
    temperature=0.1,
    convert_system_message_to_human=True,
    transport="rest"
)



@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city"""
    return f"Bad weather all the day {city}"

system_prompt = """
You are a helpful assistant that reasons step by step before answering.

"""

@tool
def get_motivation(question: str) -> str:
    """Get possitive message for a given question"""
    return f"Every thing is gonna be right about {question}"

system_prompt = """
You are a helpful assistant that reasons step by step before answering.
Always check your tools before answering

"""

# Crear el agente con response_format (NO response_schema)
agent = create_react_agent(
    model=llm,
    tools=[get_weather,get_motivation],
    prompt=system_prompt,
    #response_format=WeatherResponse 
    response_format=ToolResponse
)

# Ejecutar el agente
user_input = input("Escribe tu mensaje para el agente: ")
response = agent.invoke(
    {"messages": [{"role": "user", "content": user_input}]}
)

print(response)
# Acceder a la respuesta estructurada
print("Respuesta estructurada:")
print(response["structured_response"])

# También puedes ver el contenido final del mensaje
#print("\nContenido del mensaje final:")
#print(response)
#final = response['messages'][-1].content
#print(final)

###
#weather_structured = response["structured_response"]
#print("Condiciones del clima:", weather_structured)

## Historial mensajes.

print("RAW RESPONSE:")
print(response)
print(f"n{'-'*40}")
print(response["structured_response"])
messages = response["messages"]

# Mostrar cada mensaje limpio
for msg in messages:
    print(f"{msg.__class__.__name__}:\n{msg.content}\n{'-'*40}")


