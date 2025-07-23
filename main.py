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
        
        # First, let's check if the embeddings column exists and has data
        check_sql = "SELECT COUNT(*) FROM work_orders WHERE embeddings IS NOT NULL LIMIT 1;"
        
        with psycopg2.connect(PG_CONN) as conn, conn.cursor() as cur:
            # Check embeddings availability
            try:
                cur.execute(check_sql)
                has_embeddings = cur.fetchone()[0] > 0
            except Exception as e:
                logger.warning(f"Embeddings column might not exist: {e}")
                has_embeddings = False
            
            if has_embeddings:
                # Use semantic search with embeddings
                sql = f"""
                  SELECT {', '.join(cols)}, 1 - (embeddings <=> %s::vector) AS similarity
                  FROM work_orders
                  WHERE embeddings IS NOT NULL
                  ORDER BY embeddings <=> %s::vector
                  LIMIT 3;
                """
                cur.execute(sql, (vec, vec))
            else:
                # Fallback to text search if embeddings not available
                logger.info("Using fallback text search instead of embeddings")
                sql = f"""
                  SELECT {', '.join(cols)}, 0.5 AS similarity
                  FROM work_orders
                  WHERE description ILIKE %s
                  LIMIT 3;
                """
                cur.execute(sql, (f"%{query}%",))
            
            rows = cur.fetchall()

        results = []
        for row in rows:
            rec = dict(zip(cols + ["similarity"], row))
            # Convert Decimals to float and handle None similarities
            for k, v in rec.items():
                if isinstance(v, Decimal):
                    rec[k] = float(v)
                elif k == "similarity" and v is None:
                    rec[k] = 0.0  # Default similarity if None
            results.append(rec)

        logger.info(f"Search found {len(results)} results")
        return json.dumps(results)
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        # Return empty results instead of failing completely
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
        if not recs:
            return {
                "insights": {
                    "total": 0,
                    "avg_sim": 0.0,
                    "llm_analysis": "No relevant work orders found for your query."
                }
            }

        # Safely calculate average similarity, handling None values
        similarities = [r.get("similarity") for r in recs if r.get("similarity") is not None]
        avg = sum(similarities) / len(similarities) if similarities else 0.0

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
                "avg_sim": float(avg),
                "llm_analysis": analysis
            }
        }
    except Exception as e:
        logger.error(f"Error in compute_insights: {e}")
        # Return a user-friendly response instead of failing
        return {
            "insights": {
                "total": len(state.get("records", [])),
                "avg_sim": 0.0,
                "llm_analysis": f"Found {len(state.get('records', []))} work order(s) related to your query. Please check the detailed results."
            }
        }

def generate_response(state: ChatState) -> Dict:
    """Generate final response"""
    try:
        if state["next_action"] == "direct":
            response = LLM.invoke([HumanMessage(state["input"])]).content
            return {"response": response}
        
        i, r = state.get("insights", {}), state.get("records", [])
        if not r:
            return {"response": "No relevant work orders found for your query."}
        
        # Return structured data for card display instead of text analysis
        structured_response = {
            "type": "work_orders",
            "data": {
                "query": state["input"],
                "total_found": i.get("total", len(r)),
                "avg_similarity": i.get("avg_sim", 0.0),
                "work_orders": r[:3]  # Limit to top 3 results
            }
        }
        
        return {"response": json.dumps(structured_response)}
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        # Always provide a response to avoid frontend issues
        records = state.get("records", [])
        if records:
            return {"response": f"Found {len(records)} work orders related to your query, but encountered an issue with detailed analysis. Please try rephrasing your question."}
        else:
            return {"response": "I encountered an error processing your request. Please try again with a different query."}

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
        response_text = result.get("response", "")
        
        # Ensure we have a response
        if not response_text or response_text.strip() == "":
            response_text = "I couldn't process your request properly. Please try rephrasing your question."
            logger.warning(f"Empty response generated for input: {user_input}")
        
        logger.info(f"Generated response: {response_text[:100]}...")
        
        # Check if response is structured JSON for cards
        try:
            parsed_response = json.loads(response_text)
            if isinstance(parsed_response, dict) and parsed_response.get("type") == "work_orders":
                # Send structured data as a single message
                yield f"data: {json.dumps({'type': 'cards', 'data': parsed_response['data'], 'done': True})}\n\n"
                return
        except (json.JSONDecodeError, TypeError):
            # Not JSON, continue with text streaming
            pass
        
        # Stream the response word by word for regular text
        words = response_text.split()
        for i, word in enumerate(words):
            if i == len(words) - 1:
                yield f"data: {json.dumps({'token': word, 'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
            
    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {e}")
        error_message = "I encountered an error processing your request. Please try again with a different question."
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

@app.get("/diagnostics")
async def diagnostics():
    """Diagnostic endpoint to check database structure and embeddings"""
    try:
        diagnostics_info = {}
        
        with psycopg2.connect(PG_CONN) as conn:
            with conn.cursor() as cur:
                # Check if work_orders table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'work_orders'
                    );
                """)
                table_exists = cur.fetchone()[0]
                diagnostics_info["work_orders_table_exists"] = table_exists
                
                if table_exists:
                    # Check table structure
                    cur.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'work_orders'
                        ORDER BY ordinal_position;
                    """)
                    columns = cur.fetchall()
                    diagnostics_info["table_structure"] = dict(columns)
                    
                    # Count total records
                    cur.execute("SELECT COUNT(*) FROM work_orders;")
                    total_records = cur.fetchone()[0]
                    diagnostics_info["total_records"] = total_records
                    
                    # Check if embeddings column exists and has data
                    has_embeddings_column = "embeddings" in diagnostics_info["table_structure"]
                    diagnostics_info["has_embeddings_column"] = has_embeddings_column
                    
                    if has_embeddings_column:
                        cur.execute("SELECT COUNT(*) FROM work_orders WHERE embeddings IS NOT NULL;")
                        records_with_embeddings = cur.fetchone()[0]
                        diagnostics_info["records_with_embeddings"] = records_with_embeddings
                        
                        # Check if pgvector extension is available
                        try:
                            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                            vector_ext = cur.fetchone()
                            diagnostics_info["pgvector_extension"] = vector_ext is not None
                        except Exception as e:
                            diagnostics_info["pgvector_extension"] = False
                            diagnostics_info["pgvector_error"] = str(e)
                    else:
                        diagnostics_info["records_with_embeddings"] = 0
                        diagnostics_info["pgvector_extension"] = False
                
        return {"status": "success", "diagnostics": diagnostics_info}
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        return {"status": "error", "error": str(e)}

# ——— Run Server ——————————————————————————————————————————————————————————
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True) 