DROP TABLE IF EXISTS PATIENTS;
CREATE TABLE PATIENTS   -- containing basic patient demographic information
(
    ROW_ID INT NOT NULL PRIMARY KEY, -- Internal unique identifier for the row
    SUBJECT_ID INT NOT NULL UNIQUE, -- Unique patient identifier
    GENDER VARCHAR(5) NOT NULL, -- Gender of the patient ('m' or 'f')
    DOB TIMESTAMP(0) NOT NULL, -- Date of birth
    DOD TIMESTAMP(0) -- Date of death (if applicable)
);

DROP TABLE IF EXISTS ADMISSIONS;
CREATE TABLE ADMISSIONS -- containing hospital admission details for each patient
(
    ROW_ID INT NOT NULL PRIMARY KEY, -- Internal unique identifier
    SUBJECT_ID INT NOT NULL, -- Foreign key to PATIENTS.SUBJECT_ID
    HADM_ID INT NOT NULL UNIQUE, -- Unique hospital admission ID
    ADMITTIME TIMESTAMP(0) NOT NULL, -- Date/time of hospital admission
    DISCHTIME TIMESTAMP(0), -- Date/time of discharge
    ADMISSION_TYPE VARCHAR(50) NOT NULL, -- Type of admission (e.g., "emergency", "elective" or "urgent") (lowercase value)
    ADMISSION_LOCATION VARCHAR(50) NOT NULL, -- Where the patient was admitted from (e.g., "emergency room admit", "phys referral/normal deli", "transfer from hosp/extram", "clinic referral/premature", "transfer from skilled nur" or "trsf within this facility") (lowercase value)
    DISCHARGE_LOCATION VARCHAR(50), -- Where the patient was discharged to (e.g., "home health care", "snf", "rehab/distinct part hosp", "home", "dead/expired", "disch-tran to psych hosp", "disc-tran cancer/chldrn h", "short term hospital", "long term care hospital", "left against medical advi", "hospice-medical facility", "icf", "hospice-home" or null) (lowercase value)
    INSURANCE VARCHAR(255) NOT NULL, -- Patient's insurance provider (e.g: "medicare", "private", "medicaid", "government" or "self pay") (lowercase value)
    LANGUAGE VARCHAR(10), -- Language spoken by the patient (e.g: "engl", "russ", "ptun", "port", "span", "*pun", "cant", "gree", "*guj", "mand", "hait", "viet", "kore", "*cdi", "cape", "*ben", "ital", "arab" or null) (lowercase value)
    MARITAL_STATUS VARCHAR(50), -- Marital status of the patient (e.g: "single", "married", "widowed", "divorced", "unknown (default)", "separated" or null) (lowercase value)
    ETHNICITY VARCHAR(200) NOT NULL, -- Ethnicity of the patient (e.g: "hispanic or latino", "white", "unknown/not specified", ...etc) (lowercase value)
    AGE INT NOT NULL, -- Patient's age at time of admission
    FOREIGN KEY(SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID)
);

DROP TABLE IF EXISTS D_ICD_DIAGNOSES;
CREATE TABLE D_ICD_DIAGNOSES -- containing ICD-9 diagnosis codes
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ICD9_CODE VARCHAR(10) NOT NULL UNIQUE, -- ICD-9 diagnosis code
    SHORT_TITLE VARCHAR(50) NOT NULL, -- Short description of the diagnosis (lowercase value)
    LONG_TITLE VARCHAR(255) NOT NULL -- Full description of the diagnosis (lowercase value)
);

DROP TABLE IF EXISTS D_ICD_PROCEDURES;
CREATE TABLE D_ICD_PROCEDURES -- containing ICD-9 procedure codes
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ICD9_CODE VARCHAR(10) NOT NULL UNIQUE, -- ICD-9 procedure code
    SHORT_TITLE VARCHAR(50) NOT NULL, -- Short description of the procedure (lowercase value)
    LONG_TITLE VARCHAR(255) NOT NULL -- Full description of the procedure (lowercase value)
);

DROP TABLE IF EXISTS D_LABITEMS;
CREATE TABLE D_LABITEMS -- containing lab item metadata
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ITEMID INT NOT NULL UNIQUE, -- Unique ID for lab test
    LABEL VARCHAR(200) NOT NULL -- Name/label of the lab item (lowercase value)
);

DROP TABLE IF EXISTS D_ITEMS;
CREATE TABLE D_ITEMS -- containing all items that appear in CHARTEVENTS, INPUTEVENTS, etc.
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    ITEMID INT NOT NULL UNIQUE, -- Unique item ID
    LABEL VARCHAR(200) NOT NULL, -- Human-readable label for the item (lowercase value)
    LINKSTO VARCHAR(50) NOT NULL -- Destination table where the item is used (e.g., "chartevents", "inputevents_cv" or "outputevents") (lowercase value)
);

DROP TABLE IF EXISTS DIAGNOSES_ICD;
CREATE TABLE DIAGNOSES_ICD -- containing diagnosis records associated with hospital admissions
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL, -- Foreign key to PATIENTS
    HADM_ID INT NOT NULL, -- Foreign key to ADMISSIONS
    ICD9_CODE VARCHAR(10) NOT NULL, -- Foreign key to D_ICD_DIAGNOSES
    CHARTTIME TIMESTAMP(0) NOT NULL, -- Time the diagnosis was charted
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICD9_CODE) REFERENCES D_ICD_DIAGNOSES(ICD9_CODE)
);

DROP TABLE IF EXISTS PROCEDURES_ICD;
CREATE TABLE PROCEDURES_ICD -- containing procedure records associated with hospital admissions
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICD9_CODE VARCHAR(10) NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICD9_CODE) REFERENCES D_ICD_PROCEDURES(ICD9_CODE)
);

DROP TABLE IF EXISTS LABEVENTS;
CREATE TABLE LABEVENTS -- containing lab measurements and test results
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ITEMID INT NOT NULL, -- Foreign key to D_LABITEMS
    CHARTTIME TIMESTAMP(0),
    VALUENUM DOUBLE PRECISION, -- Numeric value of the lab test
    VALUEUOM VARCHAR(20), -- Unit of measurement
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_LABITEMS(ITEMID)
);

DROP TABLE IF EXISTS PRESCRIPTIONS;
CREATE TABLE PRESCRIPTIONS -- containing medication prescriptions and dosing information
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    STARTDATE TIMESTAMP(0) NOT NULL, -- Start of medication
    ENDDATE TIMESTAMP(0), -- End of medication
    DRUG VARCHAR(100) NOT NULL, -- Drug name (lowercase)
    DOSE_VAL_RX VARCHAR(120) NOT NULL, -- Dose value
    DOSE_UNIT_RX VARCHAR(120) NOT NULL, -- Dose unit (e.g., mg, ml)
    ROUTE VARCHAR(120) NOT NULL, -- Administration route (e.g., oral, iv) (lowercase)
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

DROP TABLE IF EXISTS COST;
CREATE TABLE COST -- containing cost data for different clinical events
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    EVENT_TYPE VARCHAR(20) NOT NULL, -- Type of event (e.g., "diagnoses_icd", "labevents", "procedures_icd", or "prescriptions") (lowercase)
    EVENT_ID INT NOT NULL, -- Foreign key to specific event table row
    CHARGETIME TIMESTAMP(0) NOT NULL,
    COST DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES DIAGNOSES_ICD(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES PROCEDURES_ICD(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES LABEVENTS(ROW_ID),
    FOREIGN KEY(EVENT_ID) REFERENCES PRESCRIPTIONS(ROW_ID)
);

DROP TABLE IF EXISTS CHARTEVENTS;
CREATE TABLE CHARTEVENTS -- containing time-series charted observations (e.g., vitals)
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    ITEMID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    VALUENUM DOUBLE PRECISION,
    VALUEUOM VARCHAR(50),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

DROP TABLE IF EXISTS INPUTEVENTS_CV;
CREATE TABLE INPUTEVENTS_CV -- containing inputs to the patient from CareVue system (e.g., IV fluids)
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    ITEMID INT NOT NULL,
    AMOUNT DOUBLE PRECISION,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

DROP TABLE IF EXISTS OUTPUTEVENTS;
CREATE TABLE OUTPUTEVENTS -- containing outputs from the patient (e.g., urine)
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    ITEMID INT NOT NULL,
    VALUE DOUBLE PRECISION,
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY(ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID),
    FOREIGN KEY(ITEMID) REFERENCES D_ITEMS(ITEMID)
);

DROP TABLE IF EXISTS MICROBIOLOGYEVENTS;
CREATE TABLE MICROBIOLOGYEVENTS -- containing microbiology culture test results
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    CHARTTIME TIMESTAMP(0) NOT NULL,
    SPEC_TYPE_DESC VARCHAR(100), -- Type of specimen (e.g., "blood culture", "urine", ...etc) (lowercase)
    ORG_NAME VARCHAR(100), -- Name of organism identified (lowercase)
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

DROP TABLE IF EXISTS ICUSTAYS;
CREATE TABLE ICUSTAYS -- containing ICU stay metadata per admission
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT NOT NULL, -- Unique ICU stay ID
    FIRST_CAREUNIT VARCHAR(20) NOT NULL, -- First ICU unit patient was admitted to (e.g: "micu", "sicu", "ccu", "tsicu", "csru") (lowercase)
    LAST_CAREUNIT VARCHAR(20) NOT NULL, -- Last ICU unit before discharge (e.g: "micu", "sicu", "ccu", "tsicu", "csru") (lowercase)
    FIRST_WARDID SMALLINT NOT NULL,
    LAST_WARDID SMALLINT NOT NULL,    
    INTIME TIMESTAMP(0) NOT NULL, -- Time of ICU admission
    OUTTIME TIMESTAMP(0), -- Time of ICU discharge
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

DROP TABLE IF EXISTS TRANSFERS;
CREATE TABLE TRANSFERS -- containing records of patient transfers between care units or wards
(
    ROW_ID INT NOT NULL PRIMARY KEY,
    SUBJECT_ID INT NOT NULL,
    HADM_ID INT NOT NULL,
    ICUSTAY_ID INT,
    EVENTTYPE VARCHAR(20) NOT NULL, -- Type of transfer (e.g., "admit", "transfer", or "discharge") (lowercase)
    CAREUNIT VARCHAR(20), -- Care unit patient was moved to/from (e.g: "micu", "sicu", "ccu", "tsicu", "csru" or null) (lowercase)
    WARDID SMALLINT,
    INTIME TIMESTAMP(0) NOT NULL,
    OUTTIME TIMESTAMP(0),
    FOREIGN KEY(HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);
