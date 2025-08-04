-- Cargar datos desde CSV dinámicamente
-- Este script cargará el archivo 'data.csv' de la carpeta LOAD/ si existe
-- Si no existe, usará el archivo por defecto de csv_samples/

-- Intentar cargar desde LOAD/data.csv primero
COPY work_orders (
    work_order_id, ac_model, aircraft_description, mel_detail_id, mel_code, mel_chapter_code,
    ata_chapter_code, issue_date, closing_date, estimated_groundtime_minutes,
    release_total_aircraft_hours, aircraft_position_issue, component_part_number,
    opco_code, workstep_text, action_text, parts_text
)
FROM '/docker-entrypoint-initdb.d/LOAD/data.csv'
DELIMITER ','
CSV HEADER; 