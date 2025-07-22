from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- 1) Estado explícito para el chatbot usando Pydantic ---
class ChatState(BaseModel):
    input: str
    action_output: Dict[str, Any] = Field(default=None)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    insights: Dict[str, float] = Field(default_factory=dict)
    response: str = Field(default="")

# --- 2) Configuración de PGVector y herramienta ---
PG_CONN = "postgresql+psycopg://user:pass@localhost:5432/mydb"
embedding = OpenAIEmbeddings()
vector_store = PGVector(
    embeddings=embedding,
    connection=PG_CONN,
    collection_name="adds_table",
    use_jsonb=True
)
retriever = vector_store.as_retriever(search_kwargs={"k":5})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="semantic_search",
    description="Busca ADDS similares en PostgreSQL+pgvector y devuelve documentos con metadata groundtime y ata_chapter."
)

# --- 3) LLM y agente ReAct ---
llm = ChatOpenAI(temperature=0)
react_agent = create_react_agent(llm, [retriever_tool])

# --- 4) Funciones/nodos del grafo ---

def decide_action(state: ChatState) -> ChatState:
    """
    Llama al agente ReAct para decidir si usar la herramienta o responder.
    """
    output = react_agent.run(state.input)
    state.action_output = output
    return state


def run_search(state: ChatState) -> ChatState:
    """
    Ejecuta semantic_search si action == 'tools'.
    """
    if state.action_output.get('action') == 'tools':
        docs = retriever_tool.invoke({"query": state.action_output.get('input', '')})
        state.records = [
            {"groundtime": float(doc.metadata.get('groundtime', 0)),
             "ata": doc.metadata.get('ata_chapter'),
             "content": doc.page_content}
            for doc in docs
        ]
    return state


def compute_insights(state: ChatState) -> ChatState:
    """
    Calcula tiempo promedio de groundtime de los registros.
    """
    if state.records:
        avg = sum(r['groundtime'] for r in state.records) / len(state.records)
    else:
        avg = 0.0
    state.insights = {'groundtime_avg': avg}
    return state


def generate_response(state: ChatState) -> ChatState:
    """
    Genera la respuesta final, bien sea la salida directa del agente o basada en insights.
    """
    action = state.action_output.get('action')
    if action == 'final':
        state.response = state.action_output.get('output', '')
    else:
        prompt = f"Insights de la búsqueda semántica: {state.insights}. ¿En qué más puedo ayudar?"
        resp = llm.chat([{ 'role': 'system', 'content': prompt }])
        state.response = resp.content
    return state

# --- 5) Construcción explícita con StateGraph ---
workflow = StateGraph(ChatState)
workflow.add_node(decide_action)
workflow.add_node('search', ToolNode([retriever_tool]))
workflow.add_node(run_search)
workflow.add_node(compute_insights)
workflow.add_node(generate_response)

# Inicio
workflow.add_edge(START, decide_action)
# Ruta: decide -> search o final
workflow.add_conditional_edges(
    decide_action,
    tools_condition,
    { 'tools': 'search', 'final': generate_response }
)
# Tras search, llamar run_search
workflow.add_edge('search', run_search)
# Luego insights
workflow.add_edge(run_search, compute_insights)
# Luego respuesta
workflow.add_edge(compute_insights, generate_response)
# Fin
workflow.add_edge(generate_response, END)

# --- 6) Compilar y ejecutar ---
graph = workflow.compile()

if __name__ == '__main__':
    initial = ChatState(input="Traeme ADDS similares a: 'ELEVATOR FAULT'")
    final_state = graph.run(initial)
    print(final_state.response)
