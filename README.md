# Lang-Lab PoC: PostgreSQL + pgvector + Embeddings

Este proyecto es una prueba de concepto (PoC) para búsquedas semánticas usando PostgreSQL con la extensión pgvector y generación de embeddings con Python.

---

## Requisitos

- Docker y Docker Compose
- Python 3.8+
- Instalar dependencias Python:
  ```sh
  pip install -r requirements.txt
  ```

---

## Uso paso a paso

Crear el venv en instalar las dependencias (Hay un freeze y el que no esta versionado)

### 1. Levantar la base de datos limpia

```sh
docker compose -f .\postgres-docker\docker-compose.yml down -v
docker compose -f .\postgres-docker\docker-compose.yml up -d
```

Esto crea la base de datos, instala la extensión pgvector y carga los datos iniciales desde los scripts y CSV.

---

### 2. Generar embeddings para los registros

```sh
python emb_gen.py
```

Este script conecta a la base, genera embeddings para los work orders y los guarda en la columna `embeddings`.

---

### 3. Configurar variables de entorno

Copia el archivo de ejemplo y configura tus variables:

```sh
cp env.example .env
```

Edita `.env` y configura:

- `GEMINI_API_KEY`: Tu clave de API de Google Gemini
- `PG_CONN`: Cadena de conexión a PostgreSQL (por defecto está configurada)

---

### 4. Ejecutar la demo interactiva (modo legacy)

```sh
python .\gemin_test_react_.py
```

Se abrirá un prompt para que escribas tu consulta. El sistema buscará los work orders más similares y generará una respuesta.

---

### 5. Ejecutar el servidor FastAPI

Para usar la API conversacional:

```sh
uvicorn main:app --host 0.0.0.0 --port 5001 --reload
```

Esto iniciará el servidor en `http://localhost:5001`. Puedes:

- Ver la documentación interactiva en: `http://localhost:5001/docs`
- Probar el health check en: `http://localhost:5001/health`
- Usar el endpoint de chat: `POST http://localhost:5001/chat`

**Ejemplo de uso del endpoint `/chat`:**

```bash
curl -X POST "http://localhost:5001/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How do I fix a leaking pipe?"}'
```

El endpoint retorna una respuesta en streaming (Server-Sent Events) que puedes consumir desde una aplicación frontend.

---

### 5. Ejecutar el servidor FastAPI

Para usar la API conversacional:

```sh
uvicorn main:app --host 0.0.0.0 --port 5001 --reload
```

Esto iniciará el servidor en `http://localhost:5001`. Puedes:

- Ver la documentación interactiva en: `http://localhost:5001/docs`
- Probar el health check en: `http://localhost:5001/health`
- Usar el endpoint de chat: `POST http://localhost:5001/chat`

**Ejemplo de uso del endpoint `/chat`:**

```bash
curl -X POST "http://localhost:5001/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How do I fix a leaking pipe?"}'
```

El endpoint retorna una respuesta en streaming (Server-Sent Events) que puedes consumir desde una aplicación frontend.

---

## Notas

- Puedes modificar los datos en `init-db/work_orders.csv` y reiniciar la base para probar con otros ejemplos.
- Si cambias la estructura de la tabla, recuerda actualizar los scripts y el generador de embeddings.
- Asegúrate de que los embeddings estén generados antes de probar la búsqueda semántica.

---

## Troubleshooting

- Si no ves resultados en la búsqueda, revisa que los embeddings estén generados y que la base tenga datos.
- Para limpiar todo y empezar de cero, usa el comando `docker compose ... down -v`.

---
