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

### 3. Ejecutar la demo interactiva

```sh
python .\gemin_test_react_.py
```

Se abrirá un prompt para que escribas tu consulta. El sistema buscará los work orders más similares y generará una respuesta.

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


