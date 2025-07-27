CREATE TABLE patients
(
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pid          VARCHAR(50)  NOT NULL,
    name         VARCHAR(255) NOT NULL,
    name_keyword VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20),
    gender       VARCHAR(20),   --options: M(male) or F(female)
    address      VARCHAR(255),
    birthday     DATE,
    allergies       TEXT,
    chronic_disease TEXT,
    report_url      TEXT,
    has_allergies   BOOLEAN NOT NULLDEFAULT false
);
CREATE UNIQUE INDEX patients_pid_index ON patients (pid);
CREATE INDEX patients_name_keyword_index ON patients (name_keyword);
CREATE INDEX patients_birthday_index patients (birthday);
----------------------------------------------------------------------------------------------------
CREATE TABLE visits
(
    id                        UUID PRIMARY KEY              DEFAULT gen_random_uuid(),
    patient_id                UUID                 NOT NULL,
    type                      VARCHAR(10)                 NOT NULL, --options: "OPD" (outpatient department - ngoại trú), "IPD" (In-patient Department - nội trú) and "ED" (Emergency Department - cấp cứu)
    location                  VARCHAR(50)                 NOT NULL, --hospital names in Vinmec Health System
    specialty                 VARCHAR(255),
    started_at                TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    ended_at                  TIMESTAMP WITHOUT TIME ZONE,
    reason                    TEXT,
    treatment_plan            TEXT,
    los                       INTEGER,	--length of stay in hospital, calculate = ended_at - started_at, unit = day
    provider_name             VARCHAR(255),
    insurance_number          VARCHAR(255),
    insurance_expired_date    DATE,
    fall_assessment_type      VARCHAR(255),
    fall_assessment_score     VARCHAR(255),
    overall_assessment_status VARCHAR(255),
    fall_assessment_status    BOOLEAN,
    medical_history           VARCHAR(255),
    disease_history           VARCHAR(255),
    disease_progression       VARCHAR(255),
    clinical_examination      VARCHAR(255),
    ihc                       VARCHAR(255),
    family_history            VARCHAR(255),
    health_related_behavior   VARCHAR(255),
    CONSTRAINT fk_visits FOREIGN KEY (patient_id) REFERENCES patients (id)
);
CREATE INDEX visits_patient_id_index ON visits (patient_id);
CREATE INDEX visits_diagnosis_index visits (diagnosis);
CREATE INDEX visits_patient_id_index visits (patient_id);
CREATE INDEX visits_provider_name_index visits (provider_name);
----------------------------------------------------------------------------------------------------
CREATE TABLE diagnosis_code 
(
    id          VARCHAR(50) NOT NULL,
    visit_id    UUID NOT NULL,
    code        VARCHAR(50),
    type        VARCHAR(50),
    name        VARCHAR(255),
    code_type   VARCHAR(50),    --2 options: "PRIMARY" (corresspond to main diagnosis/identified diagnosis) or "SECONDARY" (corresspond to secondary diagnosis/accompanying diagnosis)
    synced_at   TIMESTAMP WITHOUT TIME ZONE,
    version     INTEGER NOT NULL DEFAULT 0, 
    started_at  TIMESTAMP WITHOUT TIME ZONE,
    patient_id  UUID NOT NULL,
    CONSTRAINT fk_diagnosis_code FOREIGN KEY (visit_id) REFERENCES visits (id)
);
CREATE INDEX diagnosis_code_patient_id_index ON diagnosis_code (patient_id);
CREATE INDEX diagnosis_code_visit_id_index ON diagnosis_code (visit_id);
----------------------------------------------------------------------------------------------------
CREATE TABLE vital_signs
(
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id     UUID  NOT NULL,
    sign_type    VARCHAR(100) NOT NULL, --options: "TEMPERATURE", "WEIGHT", "RESPIRATORY_RATE", "BLOOD_PRESSURE", "HEIGHT", "PULSE" or "SATURATED_PERIPHERAL_OXYGEN"
    value_type   VARCHAR(20)  NOT NULL,	--2 options: "TEXT" or "NUMBER". If value_type=="TEXT", vital sign value is in "value_text" column else "value_number" column
    value_text   VARCHAR(255),
    value_number NUMERIC,
    unit         VARCHAR(50),   --unit of vital sign value
    reference_low     VARCHAR(255), --lower bound of the normal range of the vital sign value
    reference_high     VARCHAR(255),    --upper bound of the normal range of the vital sign value
    abnormal     BOOLEAN,
    patient_id   UUID    NOT NULL,
    param_id    VARCHAR(255),
    CONSTRAINT fk_vital_signs FOREIGN KEY (visit_id) REFERENCES visits (id)
);
CREATE INDEX vital_signs_visit_id_index ON vital_signs (visit_id);
CREATE INDEX vital_signs_encounter_at_index vital_signs (encounter_at);
CREATE INDEX vital_signs_patient_id_index vital_signs (patient_id);
----------------------------------------------------------------------------------------------------
CREATE TABLE drug_orders
(
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id      UUID NOT NULL,
    encounter_at  TIMESTAMP WITHOUT TIME ZONE,
    name          VARCHAR(255),
    unit          VARCHAR(150), --examples: "gram", "Xịt", "Gói", "Viên nang", "Miếng dán", "Viên ngậm", "Ống", "gói" , "lọ", "Viên", etc.
    dosage        VARCHAR(255),
    schedule      VARCHAR(255),
    diagnosis     VARCHAR(255),
    report_url    VARCHAR(255),
    provider_name VARCHAR(255),
    patient_id  UUID NOT NULL,
    CONSTRAINT fk_drug_orders FOREIGN KEY (visit_id) REFERENCES visits (id)
);
CREATE INDEX drug_orders_visit_id_index ON drug_orders (visit_id);
CREATE INDEX drug_orders_encounter_at_index ON drug_orders (encounter_at);
CREATE INDEX drug_orders_name_index ON drug_orders (name);
CREATE INDEX drug_orders_patient_id_index ON drug_orders (patient_id);
----------------------------------------------------------------------------------------------------
CREATE TABLE imaging_diagnosis_orders   --information of imaging diagnosis orders
(
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id      UUID NOT NULL,
    encounter_at  TIMESTAMP WITHOUT TIME ZONE,
    name          VARCHAR(255),
    modality      VARCHAR(255), --options: "BMD" (Bone Mineral Densitometry), "XA" (X-Ray Angiography), "US" (Ultrasound), "DR" (Digital Radiography), "CT" (Computed Tomography), "MG" (Mammography), "MR" (Magnetic Resonance)
    finding_type    VARCHAR(50),    --options: "XML", "HTML", "TEXT" or "MARKDOWN"
    finding         TEXT,
    impression_type VARCHAR(50),    --options: "XML", "HTML", "TEXT" or "MARKDOWN"
    impression      TEXT,
    image_url     VARCHAR(255),
    report_url    VARCHAR(255),
    provider_name VARCHAR(255),
    accession_number    VARCHAR(255),
    patient_id  UUID NOT NULL,
    CONSTRAINT fk_imaging_diagnosis_orders FOREIGN KEY (visit_id) REFERENCES visits (id)
);
CREATE INDEX imaging_diagnosis_orders_visit_id_index ON imaging_diagnosis_orders (visit_id);
CREATE INDEX imaging_diagnosis_orders_patient_id_index ON imaging_diagnosis_orders (patient_id);
----------------------------------------------------------------------------------------------------
CREATE TABLE lab_orders	--Subclinical examination test
(
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visit_id      UUID NOT NULL,
    encounter_at  TIMESTAMP WITHOUT TIME ZONE,
    name          VARCHAR(255),	--Subclinical name
    type          VARCHAR(255),	--Subclinical type
    report_url    VARCHAR(255),
    provider_name VARCHAR(255),	--doctor name
    patient_id  UUID NOT NULL,
    lab_type    VARCHAR(50) NOT NULL DEFAULT 'LAB', -- 2 options: 'LAB' or 'PAT'
    CONSTRAINT fk_lab_orders FOREIGN KEY (visit_id) REFERENCES visits (id)
);
CREATE INDEX lab_orders_visit_id_index ON lab_orders (visit_id);
CREATE INDEX lab_orders_encounter_at_index lab_orders (encounter_at);
CREATE INDEX lab_orders_name_index lab_orders (name);
CREATE INDEX lab_orders_patient_id_index lab_orders (patient_id);
----------------------------------------------------------------------------------------------------
CREATE TABLE lab_order_lines	--Subclinical indicator results
(
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lab_order_id  UUID NOT NULL,
    name          VARCHAR(255),	--Subclinical indicator name
    value_type    VARCHAR(20) NOT NULL,	--2 options: "TEXT" or "NUMBER". If value_type=="TEXT", vital sign value is in "value_text" column else "value_number" column
    value_text    VARCHAR(255),
    value_number  NUMERIC,
    unit          VARCHAR(50), --Examples:  "mmol/L", "fL", "g/L", "T/L", "fl", "L/L", "%", "pg", "G/I", "G/L", etc.
    reference_low     VARCHAR(255), --lower bound of the normal range of the value
    reference_high     VARCHAR(255),    --upper bound of the normal range of the value
    abnormal      BOOLEAN,
    display_order INTEGER     NOT NULL    DEFAULT 0,
    param_id    VARCHAR(255),
    CONSTRAINT fk_lab_order_lines FOREIGN KEY (lab_order_id) REFERENCES lab_orders (id)
);
CREATE INDEX lab_order_lines_lab_order_id_index ON lab_order_lines (lab_order_id);
