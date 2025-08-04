#!/usr/bin/env python3
import logging
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
# Buscar el archivo .env en el directorio raíz del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la base de datos desde variables de entorno
PG_CONN = os.getenv("PG_CONN", "postgresql://user:pass@localhost:5432/mydb")

def get_connection_string_for_postgres():
    """Convierte la conexión para conectarse a la base de datos postgres (default)"""
    # Parsear la conexión original
    if PG_CONN.startswith("postgresql://"):
        # Extraer partes de la conexión
        parts = PG_CONN.replace("postgresql://", "").split("@")
        if len(parts) == 2:
            auth, host_port_db = parts
            user_pass = auth.split(":")
            host_port, db_name = host_port_db.split("/")
            
            if len(user_pass) == 2:
                user, password = user_pass
                host, port = host_port.split(":")
                return f"postgresql://{user}:{password}@{host}:{port}/postgres"
    
    # Fallback
    return "postgresql://user:pass@localhost:5432/postgres"

def drop_database():
    """Elimina la base de datos especificada"""
    try:
        # Conectar a la base de datos postgres (default)
        postgres_conn = get_connection_string_for_postgres()
        logger.info(f"Conectando a postgres para eliminar la base de datos...")
        
        conn = psycopg2.connect(postgres_conn)
        conn.autocommit = True  # Necesario para DROP DATABASE
        
        with conn.cursor() as cur:
            # Obtener el nombre de la base de datos desde PG_CONN
            db_name = PG_CONN.split("/")[-1]
            logger.info(f"Eliminando base de datos: {db_name}")
            
            # Terminar todas las conexiones activas a la base de datos
            cur.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """)
            
            # Eliminar la base de datos
            cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
            logger.info(f"Base de datos '{db_name}' eliminada exitosamente")
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error al eliminar la base de datos: {e}")
        return False

def recreate_database():
    """Recrea la base de datos vacía"""
    try:
        postgres_conn = get_connection_string_for_postgres()
        logger.info("Recreando la base de datos...")
        
        conn = psycopg2.connect(postgres_conn)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            db_name = PG_CONN.split("/")[-1]
            cur.execute(f"CREATE DATABASE {db_name};")
            logger.info(f"Base de datos '{db_name}' recreada exitosamente")
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error al recrear la base de datos: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--recreate":
        logger.info("Modo: Eliminar y recrear base de datos")
        if drop_database():
            if recreate_database():
                logger.info("Proceso completado: Base de datos eliminada y recreada")
            else:
                logger.error("Error al recrear la base de datos")
                exit(1)
        else:
            logger.error("Error al eliminar la base de datos")
            exit(1)
    else:
        logger.info("Modo: Solo eliminar base de datos")
        if drop_database():
            logger.info("Proceso completado: Base de datos eliminada")
        else:
            logger.error("Error al eliminar la base de datos")
            exit(1) 