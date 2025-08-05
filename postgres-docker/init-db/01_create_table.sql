-- Crear tabla work_orders
CREATE TABLE work_orders (
    work_order_id VARCHAR(20) PRIMARY KEY,
    work_order_type VARCHAR(20),
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
    transfer_reason_text TEXT,

    -- Embeddings vectoriales
    embeddings vector(1024),

    joined_text TEXT
); 