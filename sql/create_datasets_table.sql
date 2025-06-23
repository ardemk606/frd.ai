-- Таблица для LoRA адаптеров
CREATE TABLE IF NOT EXISTS lora_adapter (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    object_name VARCHAR(500) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Таблица для задач
CREATE TABLE IF NOT EXISTS task (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL
);
-- Таблица для хранения метаданных датасетов
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    object_name VARCHAR(500) NOT NULL UNIQUE,
    size_bytes BIGINT NOT NULL,
    system_prompt_object_name VARCHAR(500),
    status VARCHAR(50) DEFAULT 'NEW',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    lora_adapter_id INTEGER REFERENCES lora_adapter(id),
    task_id INTEGER REFERENCES task(id)
);