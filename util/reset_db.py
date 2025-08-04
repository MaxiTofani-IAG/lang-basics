#!/usr/bin/env python3
import logging
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables
# Buscar el archivo .env en el directorio raíz del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    try:
        logger.info(f"Ejecutando: {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error en {description}: {e}")
        logger.error(f"Salida de error: {e.stderr}")
        return False

def reset_docker_database():
    """Reinicia completamente la base de datos de Docker"""
    # Obtener el directorio raíz del proyecto (un nivel arriba de util/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docker_dir = os.path.join(project_root, "postgres-docker")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists(docker_dir):
        logger.error(f"Directorio {docker_dir} no encontrado. Asegúrate de estar en el directorio raíz del proyecto.")
        return False
    
    logger.info("Iniciando reset completo de la base de datos Docker...")
    
    # 1. Parar y eliminar el contenedor
    if not run_command(
        f"cd {docker_dir} && docker-compose down",
        "Parando contenedor Docker"
    ):
        return False
    
    # 2. Eliminar el volumen de datos
    if not run_command(
        f"cd {docker_dir} && docker volume rm postgres-docker_db_data",
        "Eliminando volumen de datos"
    ):
        logger.warning("No se pudo eliminar el volumen (puede que no exista)")
    
    # 3. Reconstruir y levantar el contenedor
    if not run_command(
        f"cd {docker_dir} && docker-compose up -d",
        "Levantando nuevo contenedor Docker"
    ):
        return False
    
    logger.info("✓ Reset completo de la base de datos completado")
    logger.info("La base de datos se ha reinicializado con los datos originales")
    return True

def quick_reset():
    """Reset rápido: solo elimina y recrea la base de datos"""
    logger.info("Iniciando reset rápido de la base de datos...")
    
    # Importar las funciones del script drop_db.py
    try:
        import sys
        import os
        # Agregar el directorio util al path para importar drop_db
        util_dir = os.path.dirname(os.path.abspath(__file__))
        if util_dir not in sys.path:
            sys.path.insert(0, util_dir)
        from drop_db import drop_database, recreate_database
        
        if drop_database():
            if recreate_database():
                logger.info("✓ Reset rápido completado")
                return True
            else:
                logger.error("Error al recrear la base de datos")
                return False
        else:
            logger.error("Error al eliminar la base de datos")
            return False
            
    except ImportError:
        logger.error("No se pudo importar drop_db.py. Asegúrate de que el archivo existe.")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--docker":
            # Reset completo con Docker
            reset_docker_database()
        elif sys.argv[1] == "--quick":
            # Reset rápido
            quick_reset()
        else:
            print("Uso:")
            print("  python reset_db.py --docker  # Reset completo con Docker (recomendado)")
            print("  python reset_db.py --quick   # Reset rápido (solo elimina/recrea DB)")
    else:
        print("Uso:")
        print("  python reset_db.py --docker  # Reset completo con Docker (recomendado)")
        print("  python reset_db.py --quick   # Reset rápido (solo elimina/recrea DB)") 