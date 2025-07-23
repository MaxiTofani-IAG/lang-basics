#!/usr/bin/env python3
import logging
import psycopg2
from sentence_transformers import SentenceTransformer

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la base de datos
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mydb',
    'user': 'user',
    'password': 'pass'
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def generate_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    conn = connect_db()
    with conn, conn.cursor() as cur:
        cur.execute("""
            SELECT wo_id, amm, description, ground_time, part_numbers
            FROM work_orders
            WHERE embeddings IS NULL
        """)
        for wo_id, *fields in cur.fetchall():
            text = " ".join(str(f) for f in fields if f is not None)
            emb = model.encode(text).tolist()
            cur.execute(
                "UPDATE work_orders SET embeddings = %s WHERE wo_id = %s",
                (emb, wo_id)
            )
    logger.info("Embeddings generados para todos los work_orders pendientes")

if __name__ == "__main__":
    try:
        logger.info("Iniciando generación de embeddings...")
        generate_embeddings()
        logger.info("Proceso completado.")
    except Exception as e:
        logger.error("Error en el proceso: %s", e)
        exit(1)
