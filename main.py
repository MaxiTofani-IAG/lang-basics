#!/usr/bin/env python3
import os
import json
import psycopg2
from decimal import Decimal
from dotenv import load_dotenv
from typing import TypedDict, List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ——— FastAPI Configuration ——————————————————————————————————————————
app = FastAPI(
    title="Work Orders Semantic Search API",
    description="API for semantic search and conversational queries on work orders database",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Pydantic Models ——————————————————————————————————————————————————
class ChatRequest(BaseModel):
    message: str

# ——— Agent State Definition ——————————————————————————————————————————
class ChatState(TypedDict):
    input: str
    records: List[Dict]
    insights: Dict
    response: str
    next_action: str

# ——— Global Configuration ——————————————————————————————————————————————
PG_CONN = os.getenv("PG_CONN", "postgresql://user:pass@localhost:5432/mydb")
EMB = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

# ——— Semantic Search Tool ——————————————————————————————————————————————
@tool(description="Returns JSON with the most similar work_orders using pgvector")
def semantic_search(query: str) -> str:
    """Perform semantic search on work orders database"""
    try:
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
            # Convert Decimals to float
            for k, v in rec.items():
                if isinstance(v, Decimal):
                    rec[k] = float(v)
            results.append(rec)

        return json.dumps(results)
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return json.dumps([])

# ——— Workflow Nodes ——————————————————————————————————————————————————
def decide_action(state: ChatState) -> Dict:
    """Decide whether to perform search or respond directly"""
    try:
        resp = LLM.invoke([
            SystemMessage("If the message mentions searching/finding/similar/ADDs or aviation concepts respond 'search', otherwise 'direct'"),
            HumanMessage(state["input"])
        ])
        return {"next_action": "search" if "search" in resp.content.lower() else "direct"}
    except Exception as e:
        logger.error(f"Error in decide_action: {e}")
        return {"next_action": "direct"}

def run_search(state: ChatState) -> Dict:
    """Execute semantic search"""
    try:
        data = json.loads(semantic_search.invoke(state["input"]))
        logger.info(f"Search results: {data}")
        return {"records": data}
    except Exception as e:
        logger.error(f"Error in run_search: {e}")
        return {"records": []}

def compute_insights(state: ChatState) -> Dict:
    """Compute insights from search results"""
    try:
        recs = state["records"]
        avg = sum(r["similarity"] for r in recs) / len(recs) if recs else 0

        prompt = (
            f"User query: {state['input']}\n"
            f"Similar results found:\n{recs}\n"
            "Analyze the results and respond concisely (max 150 words):\n"
            "1. Most similar case.\n"
            "2. If user mentioned 'AMM', indicate which AMM repeats.\n"
            "3. If user mentioned 'part' or 'PN-', indicate which part numbers repeat or summarize those used\n"
            "4. Mention any other relevant patterns.\n"
            "5. Always calculate the average of numeric fields like man_hours (if available in records)"
        )
        
        analysis = LLM.invoke([HumanMessage(prompt)]).content

        return {
            "insights": {
                "total": len(recs),
                "avg_sim": avg,
                "llm_analysis": analysis
            }
        }
    except Exception as e:
        logger.error(f"Error in compute_insights: {e}")
        return {"insights": {"total": 0, "avg_sim": 0, "llm_analysis": "Error analyzing results"}}

def generate_response(state: ChatState) -> Dict:
    """Generate final response"""
    try:
        if state["next_action"] == "direct":
            response = LLM.invoke([HumanMessage(state["input"])]).content
            return {"response": response}
        
        i, r = state["insights"], state["records"]
        if not r:
            return {"response": "No relevant work orders found for your query."}
        
        # Use the LLM analysis as the response
        return {"response": i["llm_analysis"]}
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return {"response": "I encountered an error processing your request. Please try again."}

# ——— Build Workflow Graph ——————————————————————————————————————————————
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

# ——— Compile Workflow ——————————————————————————————————————————————————
workflow = wf.compile()

# ——— Async Response Generator ——————————————————————————————————————————
async def generate_streaming_response(user_input: str):
    """Generate streaming response for the chat endpoint"""
    try:
        initial_state: ChatState = {
            "input": user_input,
            "records": [],
            "insights": {},
            "response": "",
            "next_action": ""
        }
        
        # Execute the workflow
        result = workflow.invoke(initial_state)
        response_text = result.get("response", "No response generated")
        
        # Stream the response word by word
        words = response_text.split()
        for i, word in enumerate(words):
            if i == len(words) - 1:
                yield f"data: {json.dumps({'token': word, 'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
            
    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {e}")
        error_message = "I encountered an error processing your request. Please try again."
        yield f"data: {json.dumps({'token': error_message, 'done': True})}\n\n"

# ——— API Endpoints ——————————————————————————————————————————————————————
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Work Orders Semantic Search API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that accepts user messages and returns streaming responses
    based on semantic search of work orders database
    """
    logger.info(f"Received chat request: {request.message}")
    
    return StreamingResponse(
        generate_streaming_response(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        with psycopg2.connect(PG_CONN) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# ——— Run Server ——————————————————————————————————————————————————————————
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True) 