-- 如果表存在，则删除
DROP TABLE IF EXISTS older_study_mimiciv;
create table "older_study_mimiciv" as

-- MIMIC-IV
WITH
-- 1. 基础患者筛选与数据准备
older_study_cohort_mimiciv_0 AS (
    SELECT
        icud.subject_id,
        icud.hadm_id,
        icud.stay_id,
        icud.gender,
        icud.dod AS deathtime,
        icud.admittime,
        icud.dischtime,
        icud.first_hosp_stay,
        -- 修正：允许 admission_age 为浮点数，去除非整数转换
        icud.admission_age AS age,
        icud.race,
        icud.icu_intime AS intime,
        icud.icu_outtime AS outtime,
        -- 修正：允许 los_icu_day 为浮点数
        icud.los_icu AS los_icu_day,
        ROUND(CAST(
            (DATE_PART('day', icud.dischtime - icud.admittime) * 24 +
             DATE_PART('hour', icud.dischtime - icud.admittime)) / 1440
        AS NUMERIC), 2) AS los_hospital_day,
        ROUND(CAST(
            (DATE_PART('day', icud.icu_intime - icud.admittime) * 24 +
             DATE_PART('hour', icud.icu_intime - icud.admittime)) / 1440
        AS NUMERIC), 2) AS los_icu_admit_day,
        ad.admission_type,
        pt.anchor_year_group,
        ie.first_careunit
    FROM "icustay_detail" icud
    LEFT JOIN "admissions" ad
        ON icud.subject_id = ad.subject_id
        AND icud.hadm_id = ad.hadm_id
    LEFT JOIN "patients" pt
        ON icud.subject_id = pt.subject_id
    LEFT JOIN "icustays" ie
        ON icud.stay_id = ie.stay_id
    WHERE icud.first_icu_stay = TRUE
    -- 修正：移除对 los_icu 和 los_hospital 的强制整数转换
    AND icud.admission_age >= 18
    AND icud.los_icu >= 1
    AND icud.los_hospital >= 1
),
-- 2. 数据清洗与衍生字段
older_study_cohort_mimiciv AS (
    SELECT
        icud.subject_id, icud.hadm_id, icud.stay_id,
        CASE
            WHEN first_careunit IN ('Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)') THEN 'CCU'
            WHEN first_careunit IN ('Neuro Intermediate', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)') THEN 'NICU'
            WHEN first_careunit = 'Medical Intensive Care Unit (MICU)' THEN 'MICU'
            WHEN first_careunit = 'Medical/Surgical Intensive Care Unit (MICU/SICU)' THEN 'Med-Surg_ICU'
            WHEN first_careunit = 'Surgical Intensive Care Unit (SICU)' THEN 'SICU'
            WHEN first_careunit = 'Trauma SICU (TSICU)' THEN 'TSICU'
            ELSE NULL
        END AS first_careunit,
        intime, outtime, los_icu_day,
        admittime, dischtime, deathtime, los_hospital_day,
        CASE
            WHEN deathtime::timestamp > intime::timestamp
                AND deathtime::timestamp <= (dischtime::timestamp + INTERVAL '1 DAY') THEN 1
            ELSE 0
        END AS death_hosp,
        CAST(CEIL(EXTRACT(EPOCH FROM (deathtime::timestamp - intime::timestamp)) / 3600) AS INT) AS deathtime_icu_hour,
        los_icu_admit_day,
        CASE
            WHEN admission_type IN ('AMBULATORY OBSERVATION', 'DIRECT OBSERVATION', 'EU OBSERVATION', 'OBSERVATION ADMIT') THEN 'OBSERVATION'
            WHEN admission_type IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION') THEN 'ELECTIVE'
            WHEN admission_type IN ('DIRECT EMER.', 'EW EMER.') THEN 'EMERGENCY'
            WHEN admission_type = 'URGENT' THEN 'URGENT'
            ELSE NULL
        END AS admission_type,
        CASE
            WHEN race = 'ASIAN' THEN 'asian'
            WHEN race = 'BLACK/AFRICAN AMERICAN' THEN 'black'
            WHEN race = 'HISPANIC/LATINO' THEN 'hispanic'
            WHEN race = 'WHITE' THEN 'white'
            ELSE 'other'
        END AS ethnicity,
        anchor_year_group,
        gender, age, first_hosp_stay,
        COALESCE(ht.height, 0) AS height,
        COALESCE(wt.weight, 0) AS weight,
        CASE
            WHEN wt.weight > 0 AND ht.height > 0 THEN
                ROUND((10000 * wt.weight) / (ht.height * ht.height), 2)
            ELSE NULL
        END AS bmi
    FROM older_study_cohort_mimiciv_0 icud
    LEFT JOIN (
        SELECT stay_id, ROUND(COALESCE(weight, weight_admit), 1) AS weight
        FROM first_day_weight
        WHERE COALESCE(weight, weight_admit) BETWEEN 20 AND 400
    ) wt ON icud.stay_id = wt.stay_id
    LEFT JOIN (
        SELECT
            c.stay_id::integer AS stay_id,
            ROUND(valuenum::numeric, 1) AS height,
            ROW_NUMBER() OVER (PARTITION BY c.stay_id ORDER BY c.charttime) AS rn
        FROM chartevents c
        WHERE
            c.itemid = 226730
            AND c.valuenum IS NOT NULL
            AND c.valuenum BETWEEN 120 AND 230
    ) ht ON icud.stay_id::integer = ht.stay_id AND ht.rn = 1
)

, surgflag_info AS (
    SELECT ie.stay_id
    , MAX(CASE
        WHEN LOWER(curr_service) LIKE '%surg%' THEN 1
        WHEN curr_service = 'ORTHO' THEN 1
        ELSE 0 END) AS surgical
    FROM "icustays" ie
    LEFT JOIN "services" se
        ON ie.hadm_id = se.hadm_id
        AND se.transfertime::timestamp < (ie.intime::timestamp + INTERVAL '1 DAY')
    WHERE ie.stay_id IN (SELECT stay_id FROM older_study_cohort_mimiciv)
    GROUP BY ie.stay_id
)

, vent_info AS (
    SELECT ie.stay_id
    , MAX(CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS vent
    FROM "icustays" ie
    LEFT JOIN "ventilation" v
        ON ie.stay_id::INTEGER = v.stay_id
        AND (
            v.starttime BETWEEN CAST(ie.intime AS TIMESTAMP) AND (CAST(ie.intime AS TIMESTAMP) + INTERVAL '1' DAY)
            OR v.endtime BETWEEN CAST(ie.intime AS TIMESTAMP) AND (CAST(ie.intime AS TIMESTAMP) + INTERVAL '1' DAY)
            OR v.starttime <= CAST(ie.intime AS TIMESTAMP) AND v.endtime >= (CAST(ie.intime AS TIMESTAMP) + INTERVAL '1' DAY)
        )
        AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy', 'NonInvasiveVent')
    WHERE ie.stay_id IN (SELECT stay_id FROM older_study_cohort_mimiciv)
    GROUP BY ie.stay_id
)

, pafi_0 AS (
    SELECT ie.stay_id
    , bg.charttime
    , CASE WHEN vd.stay_id IS NULL THEN pao2fio2ratio ELSE NULL END AS pao2fio2ratio_novent
    , CASE WHEN vd.stay_id IS NOT NULL THEN pao2fio2ratio ELSE NULL END AS pao2fio2ratio_vent
    , po2 AS pao2, pco2 AS paco2, baseexcess, lactate, COALESCE(fio2, fio2_chartevents) AS fio2
    FROM "icustays" ie
    INNER JOIN "bg" bg
        ON ie.subject_id::INTEGER = bg.subject_id::INTEGER
    LEFT JOIN "ventilation" vd
        ON ie.stay_id::INTEGER = vd.stay_id
        AND bg.charttime::timestamp >= vd.starttime::timestamp
        AND bg.charttime::timestamp <= vd.endtime::timestamp
        AND vd.ventilation_status IN ('InvasiveVent', 'Tracheostomy', 'Tracheostomy')
    WHERE CAST(specimen AS VARCHAR) = 'ART.'
    AND ie.stay_id IN (SELECT stay_id FROM older_study_cohort_mimiciv)
)

, pafi AS (
    SELECT pf.stay_id
    , MIN(pao2fio2ratio_novent) AS pao2fio2ratio_novent
    , MIN(pao2fio2ratio_vent) AS pao2fio2ratio_vent
    , MIN(pao2) AS pao2_min
    , MAX(paco2) AS paco2_max
    , MAX(lactate) AS lactate_max
    , MIN(baseexcess) AS baseexcess_min
    , MAX(fio2) AS fio2_max
    FROM pafi_0 pf
    INNER JOIN older_study_cohort_mimiciv sc
        ON pf.stay_id = sc.stay_id
        AND pf.charttime::timestamp <= (sc.intime + INTERVAL '1' DAY)
        AND pf.charttime::timestamp >= (sc.intime - INTERVAL '6 HOUR')
    GROUP BY pf.stay_id
)

, lab_info_extra_0 AS (
    SELECT sc.stay_id
    , CASE
        WHEN itemid::INTEGER IN (51002, 51003, 52637) THEN 'troponin'
        WHEN itemid::INTEGER = 50960 THEN 'magnesium'
        WHEN itemid::INTEGER = 50963 THEN 'bnp'
        WHEN itemid::INTEGER IN (51244, 51688, 51245) THEN 'lymphocytes'
        WHEN itemid::INTEGER IN (51256, 51695) THEN 'neutrophils'
        ELSE NULL
    END AS label
    , le.valuenum
    FROM "labevents" le
    INNER JOIN older_study_cohort_mimiciv sc
        ON sc.subject_id = le.subject_id
        AND sc.hadm_id = le.hadm_id
    WHERE le.itemid::INTEGER IN (
        51002, 51003, 52637, 50960, 50963, 51244
        , 51688, 51245, 51256, 51695
    )
    -- 修正：允许 valuenum 为浮点数
    AND le.valuenum::NUMERIC > 0  -- 使用 NUMERIC 替代 INTEGER
    AND le.charttime::TIMESTAMP <= (sc.intime + INTERVAL '1' DAY)
    AND le.charttime::TIMESTAMP >= (sc.intime - INTERVAL '6' HOUR)
)
, lab_info_extra AS (
    SELECT stay_id
    , MAX(CASE WHEN label = 'troponin' THEN valuenum ELSE NULL END) AS troponin_max
    , MAX(CASE WHEN label = 'lymphocytes' THEN valuenum ELSE NULL END) AS lymphocytes_max
    , MIN(CASE WHEN label = 'lymphocytes' THEN valuenum ELSE NULL END) AS lymphocytes_min
    , MIN(CASE WHEN label = 'neutrophils' THEN valuenum ELSE NULL END) AS neutrophils_min
    , MAX(CASE WHEN label = 'magnesium' THEN valuenum ELSE NULL END) AS magnesium_max
    , MAX(CASE WHEN label = 'bnp' THEN valuenum ELSE NULL END) AS bnp_max
    FROM lab_info_extra_0
    GROUP BY stay_id
)

, lab_info AS (
    SELECT sc.stay_id
    , creatinine_max, bilirubin_total_max AS bilirubin_max, platelets_min AS platelet_min
    , bun_max, wbc_max, glucose_max, hematocrit_max
    , potassium_max
    , sodium_max
    , bicarbonate_min, bicarbonate_max
    , pao2fio2ratio_novent, pao2fio2ratio_vent, albumin_min
    , alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min, chloride_min
    , pao2_min, paco2_max, lactate_max, baseexcess_min, fio2_max
    , troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
    , magnesium_max, fibrinogen_min, bnp_max, aniongap_max
    FROM older_study_cohort_mimiciv sc
    LEFT JOIN "first_day_lab" lab
        ON sc.stay_id = lab.stay_id
    LEFT JOIN pafi pf
        ON sc.stay_id = pf.stay_id
    LEFT JOIN lab_info_extra labe
        ON sc.stay_id = labe.stay_id
)

, vital_gcs_uo_info AS (
    SELECT sc.stay_id, gcs_min
    , COALESCE(gcs_motor, 6) AS gcsmotor
    , COALESCE(gcs_verbal, 5) AS gcsverbal
    , COALESCE(gcs_eyes, 4) AS gcseyes
    , ROUND(CAST(vital.heart_rate_mean AS NUMERIC), 1) AS heart_rate_mean
    , ROUND(CAST(vital.mbp_mean AS NUMERIC), 1) AS mbp_mean
    , ROUND(CAST(vital.sbp_mean AS NUMERIC), 1) AS sbp_mean
    , ROUND(CAST(vital.resp_rate_mean AS NUMERIC), 1) AS resp_rate_mean
    , ROUND(CAST(vital.temperature_mean AS NUMERIC), 1) AS temperature_mean
    , CASE WHEN uo.urineoutput < 0 THEN 0 ELSE uo.urineoutput END AS urineoutput
    , vital.spo2_min
    FROM older_study_cohort_mimiciv sc
    LEFT JOIN "first_day_gcs" gcs
        ON sc.stay_id = gcs.stay_id
    LEFT JOIN "first_day_vitalsign" vital
        ON sc.stay_id::INTEGER = vital.stay_id
    LEFT JOIN "first_day_urine_output" uo
        ON sc.stay_id = uo.stay_id
)

, vaso_stg AS (
    SELECT ie.stay_id, 'norepinephrine' AS treatment, ne.vaso_rate AS rate
    FROM older_study_cohort_mimiciv ie
    INNER JOIN "norepinephrine" ne
        ON ie.stay_id = ne.stay_id
        AND ne.starttime::TIMESTAMP <= (ie.intime + INTERVAL '1 DAY')
        AND ne.endtime::TIMESTAMP > ie.intime

    UNION ALL

    SELECT ie.stay_id, 'epinephrine' AS treatment, ep.vaso_rate AS rate
    FROM older_study_cohort_mimiciv ie
    INNER JOIN "epinephrine" ep
        ON ie.stay_id = ep.stay_id
        AND ep.starttime::TIMESTAMP <= (ie.intime + INTERVAL '1 DAY')
        AND ep.endtime::TIMESTAMP > ie.intime

    UNION ALL

    SELECT ie.stay_id, 'dobutamine' AS treatment, CAST(db.vaso_rate AS NUMERIC) AS rate
    FROM older_study_cohort_mimiciv ie
    INNER JOIN "dobutamine" db
        ON ie.stay_id = db.stay_id
        AND db.starttime::TIMESTAMP <= (ie.intime + INTERVAL '1 DAY')
        AND db.endtime::TIMESTAMP > ie.intime

    UNION ALL

    SELECT ie.stay_id, 'dopamine' AS treatment, CAST(dp.vaso_rate AS NUMERIC) AS rate
    FROM older_study_cohort_mimiciv ie
    INNER JOIN "dopamine" dp
        ON ie.stay_id = dp.stay_id
        AND dp.starttime::TIMESTAMP <= (ie.intime + INTERVAL '1 DAY')
        AND dp.endtime::TIMESTAMP > ie.intime
)

, vaso_mv_info AS (
    SELECT
        ie.stay_id
        , MAX(CASE WHEN treatment = 'norepinephrine' THEN rate ELSE NULL END) AS rate_norepinephrine
        , MAX(CASE WHEN treatment = 'epinephrine' THEN rate ELSE NULL END) AS rate_epinephrine
        , MAX(CASE WHEN treatment = 'dopamine' THEN rate ELSE NULL END) AS rate_dopamine
        , MAX(CASE WHEN treatment = 'dobutamine' THEN rate ELSE NULL END) AS rate_dobutamine
    FROM older_study_cohort_mimiciv ie
    LEFT JOIN vaso_stg v
        ON ie.stay_id = v.stay_id
    GROUP BY ie.stay_id
)

, activity_info AS (
    SELECT ce.stay_id
    , MAX(CASE
        WHEN itemid = 224084 AND CAST(valuenum AS VARCHAR) IN ('Ambulate', 'Dangle') THEN 1
        WHEN itemid = 229319 AND CAST(valuenum AS VARCHAR) IN ('5 - Stand - >/= One minute', '6 - Walk - 10+ Steps', '7 - Walk - 25+ Feet', '8 - Walk - 250+ Feet') THEN 1
        WHEN itemid = 229321 AND CAST(valuenum AS VARCHAR) IN ('5 - Stand - >/= One minute', '6 - Walk - 10+ Steps', '7 - Walk - 25+ Feet', '8 - Walk - 250+ Feet') THEN 1
        WHEN itemid = 229742 AND CAST(valuenum AS VARCHAR) IN ('5 - Stand - >/= One minute', '6 - Walk - 10+ Steps', '7 - Walk - 25+ Feet', '8 - Walk - 250+ Feet') THEN 1
        ELSE 0 END) AS stand
    , MAX(CASE
        WHEN itemid = 224084 AND CAST(valuenum AS VARCHAR) IN ('Bed as Chair', 'Chair', 'Commode') THEN 1
        WHEN itemid = 229319 AND CAST(valuenum AS VARCHAR) IN ('3 - Bed - Sit at edge of bed', '4 - Chair - Transfer to chair') THEN 1
        WHEN itemid = 229321 AND CAST(valuenum AS VARCHAR) IN ('2c - Lift to chair/bed', '3 - Bed - Sit at edge of bed', '4 - Chair - Transfer to chair/bed') THEN 1
        WHEN itemid = 229742 AND CAST(valuenum AS VARCHAR) IN ('2c - Lift to chair/bed', '3 - Bed - Sit at edge of bed', '4 - Chair - Transfer to chair/bed') THEN 1
        ELSE 0 END) AS sit
    , MAX(CASE
        WHEN itemid = 224084 AND CAST(valuenum AS VARCHAR) IN ('Bedrest') THEN 1
        WHEN itemid = 229319 AND CAST(valuenum AS VARCHAR) IN ('1 - Bedrest - Only lying', '2 - Bed - Turn self / Bed activity') THEN 1
        WHEN itemid = 229321 AND CAST(valuenum AS VARCHAR) IN ('1 - Bedrest - Only lying', '2a - Passive or Active ROM', '2b - Turning in bed') THEN 1
        WHEN itemid = 229742 AND CAST(valuenum AS VARCHAR) IN ('1 - Bedrest - Only lying', '2a - Passive or Active ROM', '2b - Turning in bed') THEN 1
        ELSE 0 END) AS bed
    , MAX(CASE
        WHEN valuenum IS NOT NULL THEN 1
        ELSE 0 END) AS activity_eva_flag
    FROM "chartevents" ce
    INNER JOIN older_study_cohort_mimiciv sc
        ON ce.stay_id = sc.stay_id::INTEGER
    WHERE itemid IN (224084, 229319, 229321, 229742)
    AND ce.charttime::TIMESTAMP >= (sc.intime + INTERVAL '1' DAY)
    AND ce.charttime::TIMESTAMP <= (sc.intime - INTERVAL '1' DAY)
    GROUP BY ce.stay_id
)

, score_info AS (
    SELECT sc.stay_id
    , ap.apsiii, ap.apsiii_prob
    , sf.sofa, 1 / (1 + EXP(- (-3.3890 + 0.2439 * (sf.sofa) ))) AS sofa_prob
    , oa.oasis, oa.oasis_prob
    , sp.sapsii AS saps, sp.sapsii_prob AS saps_prob
    , cci.charlson_comorbidity_index AS cci_score
    FROM older_study_cohort_mimiciv sc
    LEFT JOIN (
        SELECT hadm_id, MAX(charlson_comorbidity_index) AS charlson_comorbidity_index
        FROM "charlson"
        GROUP BY hadm_id
    ) cci
        ON sc.hadm_id = cci.hadm_id
    LEFT JOIN "apsiii" ap
        ON sc.stay_id = ap.stay_id
    LEFT JOIN "first_day_sofa" sf
        ON sc.stay_id::INTEGER = sf.stay_id
    LEFT JOIN "oasis" oa
        ON sc.stay_id::INTEGER = oa.stay_id
    LEFT JOIN "sapsii" sp
        ON sc.stay_id::INTEGER = sp.stay_id
)

, code_status_info_0 AS (
    SELECT
        ce.stay_id,
        CAST(CEIL(EXTRACT(MINUTE FROM (ce.charttime - sc.intime)) / 60.0) AS INTEGER) AS hr,
        ce.valuenum,
        ROW_NUMBER() OVER (PARTITION BY ce.stay_id ORDER BY ce.charttime DESC) AS rn
    FROM "chartevents" ce
    INNER JOIN older_study_cohort_mimiciv sc
        ON ce.stay_id = sc.stay_id::INTEGER
    WHERE itemid IN (223758, 228687)
        AND ce.valuenum IS NOT NULL
        AND CAST(CEIL(EXTRACT(MINUTE FROM (ce.charttime - sc.intime)) / 60.0) AS INTEGER) >= -24
        AND CAST(CEIL(EXTRACT(MINUTE FROM (ce.charttime - sc.intime)) / 60.0) AS INTEGER) <= 24
)

, code_status_info AS (
    SELECT stay_id
    , CASE
        WHEN CAST(valuenum AS VARCHAR) IN ('DNAR (Do Not Attempt Resuscitation)  [DNR]', 'DNAR (Do Not Attempt Resuscitation) [DNR] / DNI', 'DNI (do not intubate)', 'DNR (do not resuscitate)', 'DNR / DNI', 'Comfort measures only') THEN 1
        ELSE 0 END AS code_status
    , CASE WHEN valuenum IS NOT NULL THEN 1 ELSE 0 END AS code_status_eva_flag
    FROM code_status_info_0
    WHERE rn = 1
)

, older_study_mimiciv_0 AS (
    SELECT sc.subject_id, sc.hadm_id, sc.stay_id
    , first_careunit, los_icu_day
    , los_hospital_day
    , death_hosp
    , CASE WHEN death_hosp = 1 THEN deathtime_icu_hour ELSE NULL END AS deathtime_icu_hour
    , CASE WHEN los_icu_admit_day < 0 THEN 0 ELSE los_icu_admit_day END AS pre_icu_los_day
    , admission_type, ethnicity, anchor_year_group
    , gender, age
    , height, weight, bmi

    , CASE
        WHEN admission_type = 'ELECTIVE' AND surgical = 1 THEN 1
        WHEN admission_type IS NULL OR surgical IS NULL THEN NULL
        ELSE 0 END AS electivesurgery

    , CASE WHEN vent = 1 THEN 1 ELSE 0 END AS vent

    , creatinine_max, bilirubin_max, platelet_min
    , bun_max, wbc_max, glucose_max, hematocrit_max
    , potassium_max, sodium_max
    , bicarbonate_min, bicarbonate_max
    , pao2fio2ratio_novent, pao2fio2ratio_vent, albumin_min
    , alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min, chloride_min
    , pao2_min, paco2_max, lactate_max, baseexcess_min, fio2_max
    , troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
    , magnesium_max, fibrinogen_min, bnp_max, aniongap_max

    , gcs_min, gcseyes, gcsmotor, gcsverbal
    , urineoutput, spo2_min, heart_rate_mean
    , mbp_mean, sbp_mean, resp_rate_mean, temperature_mean

    , CASE WHEN rate_norepinephrine > 0 THEN 1 ELSE 0 END AS norepinephrine
    , CASE WHEN rate_epinephrine > 0 THEN 1 ELSE 0 END AS epinephrine
    , CASE WHEN rate_dopamine > 0 THEN 1 ELSE 0 END AS dopamine
    , CASE WHEN rate_dobutamine > 0 THEN 1 ELSE 0 END AS dobutamine

    , CASE WHEN stand = 1 THEN 1 ELSE 0 END AS activity_stand
    , CASE WHEN sit = 1 AND (stand IS NULL OR stand = 0) THEN 1 ELSE 0 END AS activity_sit
    , CASE WHEN bed = 1 AND (stand IS NULL OR stand = 0) AND (sit IS NULL OR sit = 0) THEN 1 ELSE 0 END AS activity_bed
    , CASE WHEN activity_eva_flag = 1 THEN 1 ELSE 0 END AS activity_eva_flag

    , apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score

    , CASE WHEN cs.code_status = 1 THEN 1 ELSE 0 END AS code_status
    , CASE WHEN cs.code_status_eva_flag = 1 THEN 1 ELSE 0 END AS code_status_eva_flag

    FROM older_study_cohort_mimiciv sc
    LEFT JOIN surgflag_info si
        ON sc.stay_id = si.stay_id
    LEFT JOIN vent_info vi
        ON sc.stay_id = vi.stay_id
    LEFT JOIN lab_info lab
        ON sc.stay_id = lab.stay_id
    LEFT JOIN vital_gcs_uo_info vgu
        ON sc.stay_id = vgu.stay_id
    LEFT JOIN vaso_mv_info vm
        ON sc.stay_id = vm.stay_id
    LEFT JOIN activity_info ai
        ON sc.stay_id::integer = ai.stay_id
    LEFT JOIN score_info sci
        ON sc.stay_id = sci.stay_id
    LEFT JOIN code_status_info cs
        ON sc.stay_id::integer = cs.stay_id
    ORDER BY sc.stay_id
)

SELECT DISTINCT * FROM older_study_mimiciv_0;