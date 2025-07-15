# LangLab: Boilerplate básico para LangGraph/LangChain

Este proyecto es un punto de partida (boilerplate) para experimentar con [LangGraph](https://github.com/langchain-ai/langgraph) y [LangChain](https://github.com/langchain-ai/langchain), usando agentes ReAct y herramientas personalizadas. El objetivo es facilitar la creación de asistentes conversacionales que razonan paso a paso y pueden usar herramientas externas.

## Estructura

- [`react_basic.py`](react_basic.py): Ejemplo básico de agente ReAct con herramientas simples.
- [`react_state.py`](react_state.py): Ejemplo avanzado usando grafo de estado para manejar contexto y resumen de conversación.
- [`prompts.txt`](prompts.txt): Ejemplo de prompt de usuario.
- [`requirements.txt`](requirements.txt): Dependencias necesarias.
- [`.env`](.env): Archivo para la API Key de Gemini.

## Requisitos

- Python 3.11
- Clave de API de Google Gemini (añádela en `.env` como `GEMINI_API_KEY`)

## Instalación

```sh
pip install -r requirements.txt
```

## Uso

1. Añade tu clave de Gemini en el archivo `.env`:
    ```
    GEMINI_API_KEY="tu_clave_aquí"
    ```
2. Ejecuta el ejemplo básico:
    ```sh
    python react_basic.py
    ```
   O el ejemplo con grafo de estado:
    ```sh
    python react_state.py
    ```

3. Escribe tu mensaje cuando el agente lo solicite.

## ¿Qué hace este proyecto?

- Permite probar agentes conversacionales que razonan antes de responder.
- Incluye herramientas de ejemplo (`get_weather`, `get_motivation`) que el agente puede usar.
- Muestra cómo estructurar respuestas y acceder al historial de mensajes.
- El archivo `react_state.py` muestra cómo usar un grafo de estado para resumir y mejorar el contexto conversacional.

## Objetivo

Crear una base sencilla y funcional para desarrollar asistentes conversacionales con LangGraph/LangChain, facilitando la extensión y personalización para proyectos propios.

---

¡Puedes modificar y ampliar este boilerplate según tus
