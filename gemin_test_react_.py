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
PG_CONN = os.getenv("PG_CONN")
EMB_MODEL = os.getenv("EMB_MODEL")
EMB = HuggingFaceEmbeddings(model_name=EMB_MODEL)
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

# ——— Semantic Tool ——————————————————————————————————————————————————
@tool(description="Returns JSON with the 3 most similar work_orders using pgvector")
def semantic_search(query: str) -> str:
    vec = EMB.embed_query(query)
    cols = ["work_order_id", "ac_model", "aircraft_description", "mel_code", "mel_chapter_code", 
            "ata_chapter_code", "issue_date", "closing_date", "estimated_groundtime_minutes", 
            "release_total_aircraft_hours", "aircraft_position_issue", "component_part_number", 
            "opco_code", "workstep_text", "action_text", "parts_text"]
    sql = f"""
      SELECT {', '.join(cols)}, 1 - (embeddings <=> %s::vector) AS similarity
      FROM work_orders
      WHERE embeddings IS NOT NULL
      ORDER BY embeddings <=> %s::vector
      LIMIT 3;
    """
    with psycopg2.connect(PG_CONN) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, vec))
        rows = cur.fetchall()

    results = []
    for row in rows:
        rec = dict(zip(cols + ["similarity"], row))
        # Convertir tipos no serializables
        for k, v in rec.items():
            if isinstance(v, Decimal):
                rec[k] = float(v)
            elif hasattr(v, 'isoformat'):  # Para objetos date/datetime
                rec[k] = v.isoformat()
            elif v is None:
                rec[k] = None
        results.append(rec)

    return json.dumps(results)

# ——— Flow Nodes —————————————————————————————————————————————————
def decide_action(state: ChatState) -> Dict:
    resp = LLM.invoke([
        SystemMessage("If it mentions search/find/similar/ADDs or aeronautical concepts respond 'search', otherwise 'direct'"),
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
        f"User query: {state['input']}\n"
        f"Similar results found:\n{recs}\n"
        "Analyze the results and respond briefly and concisely (max 50 words):\n"
        "1. Most similar case.\n"
        "2. If user mentioned 'AMM', identify which AMM appears most frequently.\n"
        "3. If user mentioned 'part' or 'PN-', identify repeated part numbers or summarize unique ones used.\n"
        "4. Highlight any other relevant patterns.\n"
        "5. Always calculate averages for numeric fields like estimated_groundtime_minutes (if available in records)."
    )
    # Call LLM to get analysis
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

