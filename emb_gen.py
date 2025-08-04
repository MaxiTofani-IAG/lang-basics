#!/usr/bin/env python3
import logging
import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la base de datos desde variables de entorno
PG_CONN = os.getenv("PG_CONN")
EMB_MODEL = os.getenv("EMB_MODEL")

# Configuración de columnas a vectorizar (puedes modificar esta lista)
COLUMNS_TO_VECTORIZE = [
    'work_order_id',
    'ac_model',
    'aircraft_description', 
    'mel_code',
    'mel_chapter_code',
    'ata_chapter_code',
    'aircraft_position_issue',
    'component_part_number',
    'workstep_text',
    'action_text',
    'parts_text'
]

def setup_audit_logger():
    """Configura el logger de auditoría para archivos"""
    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/embeddings_audit_{timestamp}.log"
    
    # Configurar logger de auditoría
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)
    
    # Crear handler para archivo
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Formato para auditoría
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    audit_logger.addHandler(file_handler)
    audit_logger.propagate = False  # Evitar duplicación en consola
    
    return audit_logger, log_filename

def connect_db():
    return psycopg2.connect(PG_CONN)

def ensure_joined_text_column():
    """Asegura que existe la columna joined_text en la tabla"""
    conn = connect_db()
    with conn, conn.cursor() as cur:
        # Verificar si la columna existe
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'work_orders' AND column_name = 'joined_text'
        """)
        
        if not cur.fetchone():
            logger.info("Creando columna joined_text...")
            cur.execute("ALTER TABLE work_orders ADD COLUMN joined_text TEXT")
            logger.info("✓ Columna joined_text creada")
        else:
            logger.info("✓ Columna joined_text ya existe")

def generate_embeddings():
    model = SentenceTransformer(EMB_MODEL)
    logger.info(f"Usando modelo de embeddings: {EMB_MODEL}")
    logger.info(f"Columnas a vectorizar: {', '.join(COLUMNS_TO_VECTORIZE)}")
    
    # Configurar logger de auditoría
    audit_logger, log_filename = setup_audit_logger()
    logger.info(f"Log de auditoría: {log_filename}")
    
    # Asegurar que existe la columna joined_text
    ensure_joined_text_column()
    
    conn = connect_db()
    with conn, conn.cursor() as cur:
        # Construir la query dinámicamente basada en las columnas configuradas
        columns_str = ', '.join(COLUMNS_TO_VECTORIZE)
        query = f"""
            SELECT work_order_id, {columns_str}
            FROM work_orders
            WHERE embeddings IS NULL OR joined_text IS NULL
        """
        
        logger.info(f"Ejecutando query: {query}")
        cur.execute(query)
        
        processed_count = 0
        for row in cur.fetchall():
            work_order_id = row[0]
            
            # Combinar solo los campos que no son nulos
            text_parts = []
            null_fields = []
            
            for i, value in enumerate(row[1:], 1):
                column_name = COLUMNS_TO_VECTORIZE[i-1]
                if value is not None and str(value).strip():
                    # Formato: "campo: valor"
                    text_parts.append(f"{column_name}: {str(value).strip()}")
                else:
                    null_fields.append(column_name)
            
            # Registrar auditoría
            if null_fields:
                audit_logger.info(f"Work Order {work_order_id}: Campos nulos/vacíos: {', '.join(null_fields)}")
            else:
                audit_logger.info(f"Work Order {work_order_id}: Todos los campos tienen contenido")
            
            if text_parts:
                # Unir todos los textos con pipe como separador
                combined_text = " | ".join(text_parts)
                emb = model.encode(combined_text).tolist()
                
                # Actualizar tanto embeddings como joined_text
                cur.execute(
                    "UPDATE work_orders SET embeddings = %s, joined_text = %s WHERE work_order_id = %s",
                    (emb, combined_text, work_order_id)
                )
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Procesados {processed_count} registros...")
            else:
                audit_logger.warning(f"Work Order {work_order_id}: Sin campos válidos para vectorizar")
    
    logger.info(f"Embeddings y joined_text generados para {processed_count} work_orders")
    logger.info(f"Auditoría completa guardada en: {log_filename}")

def generate_embeddings_full():
    """Genera embeddings y joined_text para TODOS los registros (sobrescribe existentes)"""
    model = SentenceTransformer(EMB_MODEL)
    logger.info(f"Usando modelo de embeddings: {EMB_MODEL}")
    logger.info(f"Columnas a vectorizar: {', '.join(COLUMNS_TO_VECTORIZE)}")
    logger.info("⚠️  MODO FULL: Sobrescribiendo todos los registros existentes")
    
    # Configurar logger de auditoría
    audit_logger, log_filename = setup_audit_logger()
    logger.info(f"Log de auditoría: {log_filename}")
    
    # Asegurar que existe la columna joined_text
    ensure_joined_text_column()
    
    conn = connect_db()
    with conn, conn.cursor() as cur:
        # Construir la query dinámicamente basada en las columnas configuradas
        columns_str = ', '.join(COLUMNS_TO_VECTORIZE)
        query = f"""
            SELECT work_order_id, {columns_str}
            FROM work_orders
        """
        
        logger.info(f"Ejecutando query: {query}")
        cur.execute(query)
        
        processed_count = 0
        for row in cur.fetchall():
            work_order_id = row[0]
            
            # Combinar solo los campos que no son nulos
            text_parts = []
            null_fields = []
            
            for i, value in enumerate(row[1:], 1):
                column_name = COLUMNS_TO_VECTORIZE[i-1]
                if value is not None and str(value).strip():
                    # Formato: "campo: valor"
                    text_parts.append(f"{column_name}: {str(value).strip()}")
                else:
                    null_fields.append(column_name)
            
            # Registrar auditoría
            if null_fields:
                audit_logger.info(f"Work Order {work_order_id}: Campos nulos/vacíos: {', '.join(null_fields)}")
            else:
                audit_logger.info(f"Work Order {work_order_id}: Todos los campos tienen contenido")
            
            if text_parts:
                # Unir todos los textos con pipe como separador
                combined_text = " | ".join(text_parts)
                emb = model.encode(combined_text).tolist()
                
                # Actualizar tanto embeddings como joined_text (sobrescribiendo)
                cur.execute(
                    "UPDATE work_orders SET embeddings = %s, joined_text = %s WHERE work_order_id = %s",
                    (emb, combined_text, work_order_id)
                )
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Procesados {processed_count} registros...")
            else:
                audit_logger.warning(f"Work Order {work_order_id}: Sin campos válidos para vectorizar")
    
    logger.info(f"Embeddings y joined_text regenerados para {processed_count} work_orders (MODO FULL)")
    logger.info(f"Auditoría completa guardada en: {log_filename}")

def generate_joined_text_only():
    """Solo genera joined_text sin embeddings (más rápido para debugging)"""
    logger.info("Generando solo joined_text (sin embeddings)...")
    logger.info(f"Columnas a combinar: {', '.join(COLUMNS_TO_VECTORIZE)}")
    
    # Configurar logger de auditoría
    audit_logger, log_filename = setup_audit_logger()
    logger.info(f"Log de auditoría: {log_filename}")
    
    # Asegurar que existe la columna joined_text
    ensure_joined_text_column()
    
    conn = connect_db()
    with conn, conn.cursor() as cur:
        # Construir la query dinámicamente basada en las columnas configuradas
        columns_str = ', '.join(COLUMNS_TO_VECTORIZE)
        query = f"""
            SELECT work_order_id, {columns_str}
            FROM work_orders
            WHERE joined_text IS NULL
        """
        
        logger.info(f"Ejecutando query: {query}")
        cur.execute(query)
        
        processed_count = 0
        for row in cur.fetchall():
            work_order_id = row[0]
            
            # Combinar solo los campos que no son nulos
            text_parts = []
            null_fields = []
            
            for i, value in enumerate(row[1:], 1):
                column_name = COLUMNS_TO_VECTORIZE[i-1]
                if value is not None and str(value).strip():
                    # Formato: "campo: valor"
                    text_parts.append(f"{column_name}: {str(value).strip()}")
                else:
                    null_fields.append(column_name)
            
            # Registrar auditoría
            if null_fields:
                audit_logger.info(f"Work Order {work_order_id}: Campos nulos/vacíos: {', '.join(null_fields)}")
            else:
                audit_logger.info(f"Work Order {work_order_id}: Todos los campos tienen contenido")
            
            if text_parts:
                # Unir todos los textos con pipe como separador
                combined_text = " | ".join(text_parts)
                
                # Actualizar solo joined_text
                cur.execute(
                    "UPDATE work_orders SET joined_text = %s WHERE work_order_id = %s",
                    (combined_text, work_order_id)
                )
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Procesados {processed_count} registros...")
            else:
                audit_logger.warning(f"Work Order {work_order_id}: Sin campos válidos para combinar")
    
    logger.info(f"Joined_text generado para {processed_count} work_orders")
    logger.info(f"Auditoría completa guardada en: {log_filename}")

def generate_joined_text_full():
    """Genera joined_text para TODOS los registros (sobrescribe existentes)"""
    logger.info("Generando joined_text para todos los registros (MODO FULL)...")
    logger.info(f"Columnas a combinar: {', '.join(COLUMNS_TO_VECTORIZE)}")
    logger.info("⚠️  MODO FULL: Sobrescribiendo todos los registros existentes")
    
    # Configurar logger de auditoría
    audit_logger, log_filename = setup_audit_logger()
    logger.info(f"Log de auditoría: {log_filename}")
    
    # Asegurar que existe la columna joined_text
    ensure_joined_text_column()
    
    conn = connect_db()
    with conn, conn.cursor() as cur:
        # Construir la query dinámicamente basada en las columnas configuradas
        columns_str = ', '.join(COLUMNS_TO_VECTORIZE)
        query = f"""
            SELECT work_order_id, {columns_str}
            FROM work_orders
        """
        
        logger.info(f"Ejecutando query: {query}")
        cur.execute(query)
        
        processed_count = 0
        for row in cur.fetchall():
            work_order_id = row[0]
            
            # Combinar solo los campos que no son nulos
            text_parts = []
            null_fields = []
            
            for i, value in enumerate(row[1:], 1):
                column_name = COLUMNS_TO_VECTORIZE[i-1]
                if value is not None and str(value).strip():
                    # Formato: "campo: valor"
                    text_parts.append(f"{column_name}: {str(value).strip()}")
                else:
                    null_fields.append(column_name)
            
            # Registrar auditoría
            if null_fields:
                audit_logger.info(f"Work Order {work_order_id}: Campos nulos/vacíos: {', '.join(null_fields)}")
            else:
                audit_logger.info(f"Work Order {work_order_id}: Todos los campos tienen contenido")
            
            if text_parts:
                # Unir todos los textos con pipe como separador
                combined_text = " | ".join(text_parts)
                
                # Actualizar solo joined_text (sobrescribiendo)
                cur.execute(
                    "UPDATE work_orders SET joined_text = %s WHERE work_order_id = %s",
                    (combined_text, work_order_id)
                )
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Procesados {processed_count} registros...")
            else:
                audit_logger.warning(f"Work Order {work_order_id}: Sin campos válidos para combinar")
    
    logger.info(f"Joined_text regenerado para {processed_count} work_orders (MODO FULL)")
    logger.info(f"Auditoría completa guardada en: {log_filename}")

if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "--text-only":
                logger.info("Iniciando generación de joined_text...")
                logger.info(f"Conectando a: {PG_CONN}")
                generate_joined_text_only()
                logger.info("Proceso completado.")
            elif sys.argv[1] == "--text-full":
                logger.info("Iniciando generación de joined_text FULL...")
                logger.info(f"Conectando a: {PG_CONN}")
                generate_joined_text_full()
                logger.info("Proceso completado.")
            elif sys.argv[1] == "--full":
                logger.info("Iniciando generación de embeddings FULL...")
                logger.info(f"Conectando a: {PG_CONN}")
                generate_embeddings_full()
                logger.info("Proceso completado.")
            else:
                logger.error(f"Comando no reconocido: {sys.argv[1]}")
                logger.info("Comandos disponibles:")
                logger.info("  (sin argumentos) - Genera embeddings solo para registros sin embeddings/joined_text")
                logger.info("  --text-only      - Genera solo joined_text para registros sin joined_text")
                logger.info("  --text-full      - Regenera joined_text para TODOS los registros")
                logger.info("  --full           - Regenera embeddings y joined_text para TODOS los registros")
                exit(1)
        else:
            logger.info("Iniciando generación de embeddings...")
            logger.info(f"Conectando a: {PG_CONN}")
            generate_embeddings()
            logger.info("Proceso completado.")
    except Exception as e:
        logger.error("Error en el proceso: %s", e)
        exit(1)
