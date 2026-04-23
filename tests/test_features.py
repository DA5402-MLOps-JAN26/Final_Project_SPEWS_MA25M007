# tests/test_features.py 
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import (
    compute_weekly_vle_features,
    compute_demographic_features,
    compute_registration_features
)

@pytest.fixture
def sample_vle():
    return pd.DataFrame({
        'id_student': [1,1,1,2,2,3],
        'code_module': ['BBB']*6,
        'id_site': [10,11,12,10,11,10],
        'date': [7,14,21,7,14,7],
        'sum_click': [30,45,20,5,10,100],
        'week': [1,2,3,1,2,1]
    })

@pytest.fixture
def sample_info():
    return pd.DataFrame({
        'id_student': [1,2,3],
        'gender': ['M','F','M'],
        'highest_education': ['A Level or Equivalent','HE Qualification','Lower Than A Level'],
        'imd_band': ['50-60%','20-30%','80-90%'],
        'age_band': ['35-55','0-35','55<='],
        'num_of_prev_attempts': [0,1,2],
        'studied_credits': [60,120,60],
        'disability': ['N','Y','N'],
        'at_risk': [0,1,0]
    })

@pytest.fixture
def sample_reg():
    return pd.DataFrame({
        'id_student': [1,2,3],
        'code_module': ['BBB']*3,
        'date_registration': [0,0,0],
        'date_unregistration': [None, 10.0, None]
    })

def test_vle_features_shape(sample_vle):
    result = compute_weekly_vle_features(sample_vle, 2)
    assert isinstance(result, pd.DataFrame)
    assert 'id_student' in result.columns

def test_vle_cumulative_nonnegative(sample_vle):
    result = compute_weekly_vle_features(sample_vle, 3)
    assert (result['cumulative_clicks'] >= 0).all()

def test_demographic_gender_binary(sample_info):
    result = compute_demographic_features(sample_info)
    assert set(result['gender_enc'].unique()).issubset({0,1})

def test_demographic_imd_range(sample_info):
    result = compute_demographic_features(sample_info)
    assert result['imd_score'].between(1,10).all()

def test_registration_early_unreg(sample_reg):
    result = compute_registration_features(sample_reg, 4)
    assert result[result['id_student']==2]['early_unreg'].values[0] == 1

def test_assessment_features_missed_calculation():
    import pandas as pd
    from data.features import compute_assessment_features
    assess = pd.DataFrame({
        'id_student': [1, 1, 2],
        'id_assessment': [101, 102, 101],
        'date_submitted': [10.0, 20.0, 15.0],
        'score': [80.0, 90.0, 70.0]
    })
    meta = pd.DataFrame({
        'id_assessment': [101, 102],
        'date': [5, 10],
        'weight': [1.0, 2.0]
    })
    result = compute_assessment_features(assess, meta, week_num=3)
    # Student 1 should have 0 missed (both assessments submitted)
    row1 = result[result['id_student']==1]
    assert row1['missed_assessments'].values[0] == 0
    # Student 2 missed one assessment (101 submitted, 102 not)
    row2 = result[result['id_student']==2]
    assert row2['missed_assessments'].values[0] == 1

def test_assessment_features_weighted_avg():
    import pandas as pd
    from data.features import compute_assessment_features
    assess = pd.DataFrame({
        'id_student': [1, 1],
        'id_assessment': [101, 102],
        'date_submitted': [10.0, 20.0],
        'score': [80.0, 90.0]
    })
    meta = pd.DataFrame({
        'id_assessment': [101, 102],
        'date': [5, 10],
        'weight': [1.0, 3.0]
    })
    result = compute_assessment_features(assess, meta, week_num=3)
    row = result[result['id_student']==1]
    # Weighted avg = (80*1 + 90*3)/4 = 87.5
    assert abs(row['avg_weighted_score'].values[0] - 87.5) < 0.01
