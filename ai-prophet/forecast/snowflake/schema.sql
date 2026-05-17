-- Prophet Forecast — Snowflake schema
-- Run once to set up tables and Cortex Search service.

CREATE DATABASE IF NOT EXISTS PROPHET_FORECAST;
USE DATABASE PROPHET_FORECAST;
CREATE SCHEMA IF NOT EXISTS PUBLIC;

-- Main forecast log
CREATE TABLE IF NOT EXISTS FORECASTS (
    ID              NUMBER AUTOINCREMENT PRIMARY KEY,
    LOGGED_AT       TIMESTAMP_TZ,
    MARKET_ID       VARCHAR,
    QUESTION        TEXT,
    CATEGORY        VARCHAR,
    TIME_BUCKET     VARCHAR,
    SNAPSHOT_TS     TIMESTAMP_TZ,
    RESOLVES_AT     TIMESTAMP_TZ,
    P_MARKET        FLOAT,
    P_ML            FLOAT,
    P_LLM_RAW       FLOAT,
    P_CALIBRATED    FLOAT,
    P_FINAL         FLOAT,
    W_T             FLOAT,
    OUTCOME         INTEGER,      -- filled in at resolution time
    RATIONALE       TEXT,
    EVIDENCE        VARIANT,      -- JSON array of evidence dicts
    CONFIDENCE      FLOAT,
    ITERATION       INTEGER,
    BRIER           FLOAT,        -- filled in at resolution time
    ERROR           VARCHAR
);

-- Resolved forecasts view (for calibrator refit)
CREATE OR REPLACE VIEW RESOLVED_FORECASTS AS
    SELECT * FROM FORECASTS WHERE OUTCOME IS NOT NULL;

-- Cortex Search service over rationales (for analogue RAG)
-- Run after inserting resolved data:
-- CREATE OR REPLACE CORTEX SEARCH SERVICE RATIONALE_SEARCH
--     ON RATIONALE
--     ATTRIBUTES CATEGORY, TIME_BUCKET, MARKET_ID
--     WAREHOUSE = COMPUTE_WH
--     TARGET_LAG = '1 hour'
--     AS SELECT MARKET_ID, QUESTION, RATIONALE, CATEGORY, TIME_BUCKET, P_FINAL, OUTCOME
--        FROM RESOLVED_FORECASTS;
