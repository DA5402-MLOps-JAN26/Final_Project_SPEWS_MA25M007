# data/features.py 
import pandas as pd
import numpy as np
from typing import Optional

def compute_weekly_vle_features(vle_df: pd.DataFrame, week_num: int) -> pd.DataFrame:
    df = vle_df[vle_df['week'] <= week_num].copy()
    current = df[df['week'] == week_num].groupby('id_student')['sum_click'].sum().reset_index().rename(columns={'sum_click': 'weekly_clicks_current'})
    cumul = df.groupby('id_student')['sum_click'].sum().reset_index().rename(columns={'sum_click': 'cumulative_clicks'})
    last_active = df.groupby('id_student')['week'].max().reset_index().rename(columns={'week': 'last_active_week'})
    last_active['weeks_since_active'] = week_num - last_active['last_active_week']
    recent = df[df['week'] >= week_num - 2]
    weekly_agg = recent.groupby(['id_student', 'week'])['sum_click'].sum().reset_index()
    def click_slope(group):
        if len(group) < 2: return 0.0
        return float(np.polyfit(group['week'], group['sum_click'], 1)[0])
    slopes = weekly_agg.groupby('id_student').apply(click_slope).reset_index()
    slopes.columns = ['id_student', 'click_trend_slope']
    features = current.merge(cumul, on='id_student', how='outer')
    features = features.merge(last_active[['id_student', 'weeks_since_active']], on='id_student', how='outer')
    features = features.merge(slopes, on='id_student', how='outer')
    features = features.fillna(0)
    return features

def compute_assessment_features(assess_df: pd.DataFrame, assess_meta: pd.DataFrame, week_num: int) -> pd.DataFrame:
    meta_w = assess_meta[assess_meta['date'] <= week_num * 7]
    merged = assess_df.merge(meta_w[['id_assessment', 'date', 'weight']], on='id_assessment', how='inner')
    latest = merged.sort_values('date').groupby('id_student').last()[['score']].reset_index().rename(columns={'score': 'latest_score'})
    avg_score = merged.groupby('id_student').apply(lambda x: np.average(x['score'].fillna(0), weights=x['weight'].fillna(1))).reset_index()
    avg_score.columns = ['id_student', 'avg_weighted_score']
    all_due = len(meta_w['id_assessment'].unique())
    submitted = merged.groupby('id_student')['id_assessment'].nunique().reset_index().rename(columns={'id_assessment': 'submitted_count'})
    submitted['missed_assessments'] = all_due - submitted['submitted_count']
    merged['due_date'] = merged['date']
    merged['days_late'] = (merged['date_submitted'] - merged['due_date']).clip(lower=0)
    late = merged.groupby('id_student')['days_late'].mean().reset_index().rename(columns={'days_late': 'avg_days_late'})
    features = latest.merge(avg_score, on='id_student', how='outer')
    features = features.merge(submitted[['id_student', 'missed_assessments']], on='id_student', how='outer')
    features = features.merge(late, on='id_student', how='outer')
    features = features.fillna({'latest_score': 0, 'avg_weighted_score': 0, 'missed_assessments': 0, 'avg_days_late': 0})
    return features

def compute_demographic_features(info_df: pd.DataFrame) -> pd.DataFrame:
    df = info_df[['id_student', 'gender', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability']].copy()
    df['gender_enc'] = (df['gender'] == 'M').astype(int)
    df['disability_enc'] = (df['disability'] == 'Y').astype(int)
    edu_map = {'No Formal quals':0, 'Lower Than A Level':1, 'A Level or Equivalent':2, 'HE Qualification':3, 'Post Graduate Qualification':4}
    df['education_level'] = df['highest_education'].map(edu_map).fillna(1)
    imd_map = {'0-10%':1, '10-20%':2, '20-30%':3, '30-40%':4, '40-50%':5, '50-60%':6, '60-70%':7, '70-80%':8, '80-90%':9, '90-100%':10}
    df['imd_score'] = df['imd_band'].map(imd_map).fillna(5)
    return df[['id_student', 'gender_enc', 'disability_enc', 'education_level', 'imd_score', 'num_of_prev_attempts', 'studied_credits']]

def compute_registration_features(reg_df: pd.DataFrame, week_num: int) -> pd.DataFrame:
    df = reg_df[reg_df['code_module'] == 'BBB'].copy()
    df['early_unreg'] = (df['date_unregistration'].notna() & (df['date_unregistration'].astype(float) < week_num * 7)).astype(int)
    return df[['id_student', 'early_unreg']]

def build_feature_matrix(info_df, vle_df, assess_df, assess_meta, reg_df, week_num: int) -> pd.DataFrame:
    vle_df = vle_df.copy()
    vle_df['week'] = vle_df['date'] // 7
    vle_feats = compute_weekly_vle_features(vle_df, week_num)
    assess_feats = compute_assessment_features(assess_df, assess_meta, week_num)
    demo_feats = compute_demographic_features(info_df)
    reg_feats = compute_registration_features(reg_df, week_num)
    labels = info_df[['id_student', 'at_risk']].copy()
    X = labels.merge(vle_feats, on='id_student', how='left')
    X = X.merge(assess_feats, on='id_student', how='left')
    X = X.merge(demo_feats, on='id_student', how='left')
    X = X.merge(reg_feats, on='id_student', how='left')
    X = X.fillna(0)
    print(f"Feature matrix for week {week_num}: {X.shape}")
    return X
