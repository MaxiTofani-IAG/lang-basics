#!/usr/bin/env python3
import logging
import os
import shutil
import sys
from dotenv import load_dotenv

# Load environment variables
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def list_available_csv_files():
    """Lista todos los archivos CSV disponibles en csv_samples/"""
    csv_samples_dir = os.path.join(project_root, "csv_samples")
    csv_files = []
    
    if os.path.exists(csv_samples_dir):
        for file in os.listdir(csv_samples_dir):
            if file.endswith('.csv'):
                csv_files.append(file)
    
    return sorted(csv_files)

def copy_csv_to_load(csv_filename):
    """Copia un archivo CSV específico a la carpeta LOAD/"""
    csv_samples_dir = os.path.join(project_root, "csv_samples")
    load_dir = os.path.join(project_root, "postgres-docker", "init-db", "LOAD")
    
    source_file = os.path.join(csv_samples_dir, csv_filename)
    target_file = os.path.join(load_dir, csv_filename)
    
    # Verificar que el archivo fuente existe
    if not os.path.exists(source_file):
        logger.error(f"Archivo {csv_filename} no encontrado en csv_samples/")
        return False
    
    # Crear la carpeta LOAD si no existe
    os.makedirs(load_dir, exist_ok=True)
    
    # Limpiar archivos CSV existentes en LOAD/
    for existing_file in os.listdir(load_dir):
        if existing_file.endswith('.csv'):
            os.remove(os.path.join(load_dir, existing_file))
            logger.info(f"Archivo anterior eliminado: {existing_file}")
    
    # Copiar el nuevo archivo con nombre 'data.csv'
    target_file = os.path.join(load_dir, "data.csv")
    try:
        shutil.copy2(source_file, target_file)
        logger.info(f"✓ Archivo {csv_filename} copiado a LOAD/data.csv exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error al copiar {csv_filename}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python util/load_csv.py list                    # Listar archivos CSV disponibles")
        print("  python util/load_csv.py <nombre_archivo.csv>    # Cargar archivo específico")
        print("\nArchivos CSV disponibles:")
        
        csv_files = list_available_csv_files()
        if csv_files:
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file}")
        else:
            print("  No se encontraron archivos CSV en csv_samples/")
        return
    
    if sys.argv[1] == "list":
        csv_files = list_available_csv_files()
        if csv_files:
            print("Archivos CSV disponibles:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file}")
        else:
            print("No se encontraron archivos CSV en csv_samples/")
        return
    
    csv_filename = sys.argv[1]
    
    # Verificar que el archivo existe
    csv_files = list_available_csv_files()
    if csv_filename not in csv_files:
        logger.error(f"Archivo {csv_filename} no encontrado en csv_samples/")
        logger.info("Archivos disponibles:")
        for file in csv_files:
            logger.info(f"  - {file}")
        return
    
    # Copiar el archivo
    if copy_csv_to_load(csv_filename):
        logger.info("✓ Archivo preparado para la próxima inicialización de la base de datos")
        logger.info("Ejecuta 'python util/reset_db.py --docker' para reiniciar la DB con el nuevo archivo")
    else:
        logger.error("✗ Error al preparar el archivo")

if __name__ == "__main__":
    main() 