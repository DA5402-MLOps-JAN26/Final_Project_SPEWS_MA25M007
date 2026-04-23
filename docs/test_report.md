# Test Report – SPEWS

## Summary
tests/test_predictor.py       59      1    98%   34
--------------------------------------------------------
TOTAL                        429     41    90%
Coverage HTML written to dir htmlcov
============================== 21 passed in 2.00s ==============================

## Test Results
tests/test_api.py::test_health PASSED                                    [  4%]
tests/test_api.py::test_ready PASSED                                     [  9%]
tests/test_api.py::test_model_info PASSED                                [ 14%]
tests/test_api.py::test_predict PASSED                                   [ 19%]
tests/test_api.py::test_predict_batch PASSED                             [ 23%]
tests/test_api.py::test_metrics PASSED                                   [ 28%]
tests/test_api.py::test_metrics_content PASSED                           [ 33%]
tests/test_features.py::test_vle_features_shape PASSED                   [ 38%]
tests/test_features.py::test_vle_cumulative_nonnegative PASSED           [ 42%]
tests/test_features.py::test_demographic_gender_binary PASSED            [ 47%]
tests/test_features.py::test_demographic_imd_range PASSED                [ 52%]
tests/test_features.py::test_registration_early_unreg PASSED             [ 57%]
tests/test_features.py::test_assessment_features_missed_calculation PASSED [ 61%]
tests/test_features.py::test_assessment_features_weighted_avg PASSED     [ 66%]
tests/test_monitoring.py::test_update_drift_metrics PASSED               [ 71%]
tests/test_monitoring.py::test_update_model_metrics PASSED               [ 76%]
tests/test_predictor.py::test_fallback_model_creation PASSED             [ 80%]
tests/test_predictor.py::test_predict_single_with_valid_features PASSED  [ 85%]
tests/test_predictor.py::test_predict_single_risk_level_thresholds PASSED [ 90%]
tests/test_predictor.py::test_top_features_fallback_importance PASSED    [ 95%]
tests/test_predictor.py::test_predict_batch PASSED                       [100%]

## Coverage Report
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
api/__init__.py                0      0   100%
api/main.py                   59     10    83%   22-26, 64, 82-84, 89
api/predictor.py              58     15    74%   24-35, 71, 77, 84
api/schemas.py                55      0   100%
data/features.py              80     15    81%   57, 108-122
monitoring/exporter.py        14      0   100%
tests/test_api.py             47      0   100%
tests/test_features.py        49      0   100%
tests/test_monitoring.py       8      0   100%
tests/test_predictor.py       59      1    98%   34
--------------------------------------------------------
TOTAL                        429     41    90%
Coverage HTML written to dir htmlcov
============================== 21 passed in 2.00s ==============================

## Conclusion
All tests passed. Code coverage exceeds 80% threshold. Acceptance criteria fully satisfied.
