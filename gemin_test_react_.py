import os, json, psycopg2
from decimal import Decimal
from dotenv import load_dotenv
from typing import TypedDict, List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

# ——— Estado tipado ——————————————————————————————————————————————————
class ChatState(TypedDict):
    input: str
    records: List[Dict]
    insights: Dict
    response: str
    next_action: str

# ——— Configuración ——————————————————————————————————————————————————
load_dotenv()
PG_CONN = os.getenv("PG_CONN", "postgresql://user:pass@localhost:5432/mydb")
EMB = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

# ——— Tool semántica ——————————————————————————————————————————————————
@tool(description="Devuelve JSON con los 5 work_orders más similares usando pgvector")
def semantic_search(query: str) -> str:
    vec = EMB.embed_query(query)
    cols = ["wo_id","technician","opco","amm","description","ground_time","man_hours","part_numbers"]
    sql = f"""
      SELECT {', '.join(cols)}, 1 - (embeddings <=> %s::vector) AS similarity
      FROM work_orders
      ORDER BY embeddings <=> %s::vector
      LIMIT 3;
    """
    with psycopg2.connect(PG_CONN) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, vec))
        rows = cur.fetchall()

    results = []
    for row in rows:
        rec = dict(zip(cols + ["similarity"], row))
        # Convertir Decimals a float
        for k, v in rec.items():
            if isinstance(v, Decimal):
                rec[k] = float(v)
        results.append(rec)

    return json.dumps(results)

# ——— Nodos del flujo —————————————————————————————————————————————————
def decide_action(state: ChatState) -> Dict:
    resp = LLM.invoke([
        SystemMessage("Si menciona buscar/encontrar/similares/ADDs o conceptos aeronautica responde 'search', sino 'direct'"),
        HumanMessage(state["input"])
    ])
    return {"next_action": "search" if "search" in resp.content.lower() else "direct"}

def run_search(state: ChatState) -> Dict:
    data = json.loads(semantic_search.invoke(state["input"]))
    print(data)
    return {"records": data}

def compute_insights(state: ChatState) -> Dict:
    recs = state["records"]
    avg = sum(r["similarity"] for r in recs) / len(recs) if recs else 0

    # Prepara un resumen de los records para el LLM
   # resumen = "\n".join(
       # f"- WO: {r['wo_id']}, Desc: {r['description']}, Parts: {r['part_numbers']}" for r in recs
   # )
    prompt = (
        f"Consulta del usuario: {state['input']}\n"
        f"Resultados similares encontrados:\n{recs}\n"
        "Analiza los resultados y responde de forma breve y concisa (máx. 50 palabras):\n"
        "1. Caso más similar.\n"
        "2. Solo si el usuario mencionó “AMM”, indica qué AMM se repite.\n"
        "3. Solo si el usuario mencionó “part” o “PN-”, indica qué part numbers se repiten y si no repetidos haz un resumen de los utilizados\n"
        "4. Menciona cualquier otro patrón relevante.\n"
        "5. Siempre calcula el promedio entre registros de los campos numericos, como man_hours (si los tienes en los registros)"
    )
    # Llama al LLM para obtener el análisis
    analysis = LLM.invoke([HumanMessage(prompt)]).content

    return {
        "insights": {
            "total": len(recs),
            "avg_sim": avg,
            "llm_analysis": analysis
        }
    }

def generate_response(state: ChatState) -> Dict:
    if state["next_action"] == "direct":
        return {"response": LLM.invoke([HumanMessage(state["input"])]).content}
    i, r = state["insights"], state["records"]
    if not r:
        return {"response": "No se encontraron registros."}
    return {"response": f"Encontré {i['total']} WOs (sim={i['avg_sim']:.3f})."}

# ——— Construcción del grafo —————————————————————————————————————————————
wf = StateGraph(ChatState)
wf.add_node("decide", decide_action)
wf.add_node("search", run_search)
wf.add_node("compute", compute_insights)
wf.add_node("respond", generate_response)
wf.add_edge(START, "decide")
wf.add_conditional_edges("decide", lambda s: s["next_action"], {"search":"search","direct":"respond"})
wf.add_edge("search","compute")
wf.add_edge("compute","respond")
wf.add_edge("respond",END)

# ——— Ejecución con estado inicial correcto —————————————————————————————
def run_workflow(user_input: str):
    initial_state: ChatState = {
        "input": user_input,
        "records": [],
        "insights": {},
        "response": "",
        "next_action": ""
    }
    program = wf.compile()
    return program.invoke(initial_state)

# ——— Uso —————————————————————————————————————————————————————————————
if __name__ == "__main__":
    user_input = input("Ingrese su consulta: ")
    salida = run_workflow(user_input)
    print(salida["response"])
    print(salida["insights"])

