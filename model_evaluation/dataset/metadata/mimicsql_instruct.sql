create table DEMOGRAPHIC    -- Contains patient-level demographic and admission data.
(
    SUBJECT_ID         TEXT,   -- Unique patient identifier
    HADM_ID            TEXT,   -- Unique hospital admission identifier
    NAME               TEXT,   -- Patient's name
    MARITAL_STATUS     TEXT,   -- Marital status of the patient ("SINGLE", "DIVORCED", "MARRIED", "WIDOWED", "UNKNOWN (DEFAULT)", or null) (uppercase value)
    AGE                INTEGER,-- Age of the patient at admission
    DOB                TEXT,   -- Date of birth
    GENDER             TEXT,   -- Gender of the patient, "M" (male) or "F" (female)
    LANGUAGE           TEXT,   -- Primary language spoken by the patient: "POLI", "ENGL" (english), "MAND", "SPAN" (spanish), "VIET" (vietnamese), "RUSS" (russian) (uppercase value)
    RELIGION           TEXT,   -- Patient's reported religion: "CATHOLIC", "OTHER", "PROTESTANT QUAKER", "JEWISH", "UNOBTAINABLE", "CHRISTIAN SCIENTIST", "BUDDHIST", "ROMANIAN EAST. ORTH", "MUSLIM" or "NOT SPECIFIED" (some cases are null) (uppercase value)
    ADMISSION_TYPE     TEXT,   -- ype of hospital admission: "EMERGENCY", "ELECTIVE" or "URGENT" (uppercase value)
    DAYS_STAY          INTEGER,-- Length of stay in days
    INSURANCE          TEXT,   -- Insurance type: "Medicare", "Private", "Medicaid" or "Government" (uppercase value)
    ETHNICITY          TEXT,   -- Ethnic background of the patient (e.g: "BLACK/AFRICAN AMERICAN", "UNKNOWN/NOT SPECIFIED", "WHITE", "OTHER", "ASIAN", "HISPANIC OR LATINO", "HISPANIC/LATINO - PUERTO RICAN", "UNABLE TO OBTAIN", "AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE") (uppercase value)
    EXPIRE_FLAG        INTEGER,-- 1 if patient died during hospitalization, 0 otherwise
    ADMISSION_LOCATION TEXT,   -- Where the patient was admitted from (e.g., EMERGENCY ROOM ADMIT", "TRANSFER FROM HOSP/EXTRAM", "PHYS REFERRAL/NORMAL DELI", "CLINIC REFERRAL/PREMATURE" or "TRANSFER FROM SKILLED NUR") (uppercase value)
    DISCHARGE_LOCATION TEXT,   -- Where the patient was discharged to (e.g., "HOME HEALTH CARE", "DEAD/EXPIRED", "SNF", "REHAB/DISTINCT PART HOSP", "HOME", "HOSPICE-HOME", "DISCH-TRAN TO PSYCH HOSP", "HOME WITH HOME IV PROVIDR", "LONG TERM CARE HOSPITAL", "ICF") (uppercase value)
    DIAGNOSIS          TEXT,   -- Free-text primary diagnosis (uppercase value)
    DOD                TEXT,   -- Date of death
    DOB_YEAR           INTEGER,-- Year of birth
    DOD_YEAR           REAL,   -- Year of death (may include decimals if approximate)
    ADMITTIME          TEXT,   -- Date and time of hospital admission
    DISCHTIME          TEXT,   -- Date and time of hospital discharge
    ADMITYEAR          INTEGER -- Year of admission
);
----------------------------------------------------------------------------------------------------
create table DIAGNOSES  -- Contains diagnosis codes and their descriptions associated with admissions.
(
    SUBJECT_ID  TEXT,   -- Unique patient identifier
    HADM_ID     TEXT,   -- Hospital admission identifier
    ICD9_CODE   TEXT,   -- ICD-9 diagnosis code
    SHORT_TITLE TEXT,   -- Abbreviated description of the diagnosis
    LONG_TITLE  TEXT    -- Full description of the diagnosis
);
----------------------------------------------------------------------------------------------------
create table LAB    -- Contains lab test results and related metadata.
(
    SUBJECT_ID TEXT,    -- Unique patient identifier
    HADM_ID    TEXT,    -- Hospital admission identifier
    ITEMID     TEXT,    -- Identifier for the lab test
    CHARTTIME  TEXT,    -- Time when the lab result was charted
    FLAG       TEXT,    -- Indicator (e.g., abnormal, high, low)
    VALUE_UNIT TEXT,    -- Unit of the lab value (e.g., mg/dL)
    LABEL      TEXT,    -- Name of the lab test
    FLUID      TEXT,    -- Fluid type (e.g., "Blood", "Urine", "Other Body Fluid", "Ascites", "Cerebrospinal Fluid (CSF)", "Pleural", "Joint Fluid")
    CATEGORY   TEXT     -- Category of the lab test (e.g., "Blood Gas", "Chemistry" or "Hematology")
);
----------------------------------------------------------------------------------------------------
create table PRESCRIPTIONS  -- Contains information on medications ordered during hospital stay.
(
    SUBJECT_ID        TEXT,   -- Unique patient identifier
    HADM_ID           TEXT,   -- Hospital admission identifier
    ICUSTAY_ID        TEXT,   -- ICU stay identifier (if applicable)
    DRUG_TYPE         TEXT,   -- Type of drug (e.g., "MAIN", "BASE" or "ADDITIVE") (uppercase value)
    DRUG              TEXT,   -- Name of the medication
    FORMULARY_DRUG_CD TEXT,   -- Internal hospital code for the drug (uppercase value)
    ROUTE             TEXT,   -- Route of administration (e.g., "IM", "PO", "IV", "IH", "ORAL", ...) (uppercase value)
    DRUG_DOSE         TEXT    -- Dosage of the drug
);
----------------------------------------------------------------------------------------------------
create table PROCEDURES -- Contains procedure codes and descriptions associated with admissions.
(
    SUBJECT_ID  TEXT,   -- Unique patient identifier
    HADM_ID     TEXT,   -- Hospital admission identifier
    ICD9_CODE   TEXT,   -- ICD-9 procedure code
    SHORT_TITLE TEXT,   -- Abbreviated description of the procedure
    LONG_TITLE  TEXT    -- Full description of the procedure
);