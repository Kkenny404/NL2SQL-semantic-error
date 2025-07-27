-- This query fundamentally misinterprets the question by counting all patients with at least one medication request, regardless of diagnosis or deceased status.
SELECT COUNT(DISTINCT P.id)
FROM `bigquery-public-data.fhir_synthea.patient` P
JOIN `bigquery-public-data.fhir_synthea.medication_request` MR
  ON MR.patientId = P.id
WHERE MR.status = 'active'