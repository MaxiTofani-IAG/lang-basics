CREATE TABLE work_orders (
    wo_id VARCHAR(20) PRIMARY KEY,
    technician VARCHAR(100) NOT NULL,
    opco VARCHAR(50) NOT NULL,
    amm VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    ground_time VARCHAR(20) NOT NULL,
    man_hours NUMERIC(4,2) NOT NULL,
    part_numbers JSONB NOT NULL,
    embeddings vector(384)
);

COPY work_orders (wo_id, technician,opco,amm, description, ground_time, man_hours, part_numbers)
FROM '/docker-entrypoint-initdb.d/work_orders.csv'
DELIMITER ','
CSV HEADER;