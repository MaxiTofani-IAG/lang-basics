# Lang-Lab PoC: PostgreSQL + pgvector + Embeddings

Prueba de concepto para búsquedas semánticas usando PostgreSQL con pgvector y embeddings generados con Python.

---

## Carga de datos

### Estructura de datos

El proyecto carga datos de work orders desde un archivo CSV. La estructura de la tabla es:

```sql
work_orders (
    work_order_id VARCHAR(20) PRIMARY KEY,
    ac_model VARCHAR(50),
    aircraft_description TEXT,
    mel_detail_id VARCHAR(50),
    mel_code VARCHAR(50),
    mel_chapter_code VARCHAR(20),
    ata_chapter_code VARCHAR(20),
    issue_date DATE,
    closing_date DATE,
    estimated_groundtime_minutes INTEGER,
    release_total_aircraft_hours NUMERIC(12,2),
    aircraft_position_issue VARCHAR(100),
    component_part_number VARCHAR(100),
    opco_code VARCHAR(10),
    workstep_text TEXT,
    action_text TEXT,
    parts_text TEXT,
    embeddings vector(1024),
    joined_text TEXT
)
```

### Archivo de datos

Para cargar tus propios datos:

1. **Coloca tu archivo CSV** en: `postgres-docker/init-db/LOAD/data.csv`
2. **Formato requerido**: CSV con headers que coincidan con las columnas de la tabla
3. **Ejemplo de estructura**:
   ```csv
   work_order_id,ac_model,aircraft_description,mel_detail_id,mel_code,mel_chapter_code,ata_chapter_code,issue_date,closing_date,estimated_groundtime_minutes,release_total_aircraft_hours,aircraft_position_issue,component_part_number,opco_code,workstep_text,action_text,parts_text
   11779023,A320,EI-DEK AIRBUS A320-214,,,,May-00,3/24/2025,3/25/2025,0,3166669.0,,,EI,"WORKSTEPS DESCRIPTIONS:...","ACTIONS:...",
   ```

### Crear el DDL de la tabla

Para que tu CSV funcione correctamente, necesitas crear un DDL que coincida con la estructura de tus datos:

1. **Analiza tu CSV**: Revisa los headers y tipos de datos
2. **Crea el archivo DDL**: Modifica `postgres-docker/init-db/01_create_table.sql`
3. **Mapea los tipos de datos**:
   ```sql
   -- Ejemplo de mapeo de tipos
   VARCHAR(20)     -- Para IDs cortos
   VARCHAR(50)     -- Para códigos y modelos
   VARCHAR(100)    -- Para descripciones largas
   TEXT            -- Para contenido extenso (workstep_text, action_text)
   DATE            -- Para fechas
   INTEGER         -- Para números enteros
   NUMERIC(12,2)   -- Para números decimales
   ```

4. **Estructura base** (copia y adapta):
   ```sql
   CREATE TABLE work_orders (
       work_order_id VARCHAR(20) PRIMARY KEY,
       ac_model VARCHAR(50),
       aircraft_description TEXT,
       -- ... tus columnas aquí ...
       
       -- Columnas para embeddings (requeridas)
       embeddings vector(1024),
       joined_text TEXT
   );
   ```

5. **Actualiza el script de carga**: Modifica `02_load_data.sql` para que coincida con tu estructura

### Archivos de ejemplo

- **Datos de prueba**: `csv_samples/work_orders_filtered_sample_*.csv`
- **Datos completos**: `csv_samples/work_orders_filtered.csv`

---

## Requisitos

- Docker y Docker Compose
- Python 3.8+
- Dependencias: `pip install -r requirements.txt`

---

## Configuración rápida

### 1. Levantar base de datos

```sh
docker compose -f .\postgres-docker\docker-compose.yml down -v
docker compose -f .\postgres-docker\docker-compose.yml up -d
```

### 2. Configurar variables de entorno

```sh
cp env.example .env
```

Edita `.env` con tu `GEMINI_API_KEY` y configuración de PostgreSQL.

### 3. Generar embeddings

```sh
python emb_gen.py
```

---

## Comandos disponibles

### Generación de embeddings

```sh
# Genera embeddings solo para registros sin embeddings/joined_text
python emb_gen.py

# Regenera embeddings y joined_text para TODOS los registros (sobrescribe)
python emb_gen.py --full

# Genera solo joined_text para registros sin joined_text (más rápido)
python emb_gen.py --text-only

# Regenera joined_text para TODOS los registros (sobrescribe)
python emb_gen.py --text-full
```

### Servidor API

```sh
# Iniciar servidor FastAPI
uvicorn main:app --host 0.0.0.0 --port 5001 --reload
```

- **Documentación**: `http://localhost:5001/docs`
- **Health check**: `http://localhost:5001/health`
- **Chat endpoint**: `POST http://localhost:5001/chat`

---

## Uso de la API

### Ejemplo de chat

```bash
curl -X POST "http://localhost:5001/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How do I fix a leaking pipe?"}'
```

### Demo interactiva (legacy)

```sh
python .\gemin_test_react_.py
```

---

## Estructura del proyecto

```
lang-basics/
├── emb_gen.py              # Generador de embeddings
├── main.py                 # API FastAPI
├── gemin_test_react_.py    # Demo interactiva
├── postgres-docker/        # Configuración de BD
├── csv_samples/           # Datos de ejemplo
└── util/                  # Utilidades de BD
```

---

## Troubleshooting

- **Sin resultados**: Verifica que los embeddings estén generados
- **Reiniciar todo**: `docker compose ... down -v`
- **Logs de auditoría**: Revisa `logs/embeddings_audit_*.log`

---

## Utilidades de base de datos

### Reset de base de datos

```sh
# Reset completo con Docker (recomendado)
python util/reset_db.py --docker

# Reset rápido (solo elimina/recrea DB)
python util/reset_db.py --quick
```

**Diferencias:**
- **`--docker`**: Para y elimina el contenedor Docker, elimina volúmenes, y reinicia todo desde cero
- **`--quick`**: Solo elimina y recrea la base de datos (más rápido, mantiene Docker)

### Eliminar base de datos

```sh
# Eliminar y recrear base de datos
python util/drop_db.py --recreate
```

### Cargar datos desde CSV

```sh
# Cargar datos desde archivo CSV
python util/load_csv.py
```

### Dividir archivos CSV grandes

```sh
# Dividir CSV en archivos más pequeños
python util/csv_splitter.py
```
