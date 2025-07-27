SELECT
  "StudyInstanceUID"
FROM
  IDC.IDC_V17.DICOM_PIVOT AS "dicom_pivot"
WHERE
  LOWER("dicom_pivot"."SegmentedPropertyTypeCodeSequence") LIKE LOWER('15825003')
  AND "dicom_pivot"."collection_id" IN ('Community', 'nsclc_radiomics')
GROUP BY
  "StudyInstanceUID";