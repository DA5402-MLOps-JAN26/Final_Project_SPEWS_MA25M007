import pandas as pd
import numpy as np
import json
import os

# Load data
assess = pd.read_csv('data/raw/studentAssessment.csv')
assessments_meta = pd.read_csv('data/raw/assessments.csv')
info = pd.read_csv('data/processed/student_labels.csv')
reg = pd.read_csv('data/raw/studentRegistration.csv')
vle = pd.read_csv('data/raw/studentVle.csv')
vle = vle[vle['code_module'] == 'BBB']

print("=== studentAssessment ===")
print(assess.shape)
print(assess['score'].describe())
print(f"Missing scores (no submission): {assess['score'].isnull().sum()}")

# Score distribution by outcome
merged = assess.merge(info[['id_student', 'at_risk', 'final_result']], on='id_student', how='left')
print("\nMean score by outcome:")
print(merged.groupby('final_result')['score'].mean())

# Late submission analysis
merged2 = assess.merge(assessments_meta[['id_assessment', 'date']], on='id_assessment', how='left')
merged2['date_submitted'] = pd.to_numeric(merged2['date_submitted'], errors='coerce')
merged2['date'] = pd.to_numeric(merged2['date'], errors='coerce')
merged2['days_late'] = merged2['date_submitted'] - merged2['date']
print("\nDays late stats:")
print(merged2['days_late'].describe())

# Compute drift baselines
vle['week'] = vle['date'] // 7
weekly = vle.groupby(['id_student', 'week'])['sum_click'].sum().reset_index()

baselines = {}
for col in ['sum_click']:
    vals = vle[col].dropna().values
    baselines[f'vle_{col}'] = {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'p25': float(np.percentile(vals, 25)),
        'p75': float(np.percentile(vals, 75)),
        'p95': float(np.percentile(vals, 95)),
    }

for col in ['score']:
    vals = assess[col].dropna().values
    baselines[f'assessment_{col}'] = {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'p25': float(np.percentile(vals, 25)),
        'p75': float(np.percentile(vals, 75)),
    }

# Early unregistration rate
reg_bbb = reg[reg['code_module'] == 'BBB'].copy()
reg_bbb['date_unregistration'] = pd.to_numeric(reg_bbb['date_unregistration'], errors='coerce')
early_unreg = reg_bbb[reg_bbb['date_unregistration'].notna() & (reg_bbb['date_unregistration'] < 30)]
baselines['early_unregistration_rate'] = float(len(early_unreg) / len(reg_bbb))

# Save baselines
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/baseline_stats.json', 'w') as f:
    json.dump(baselines, f, indent=2)

print("\nDrift baselines saved to data/processed/baseline_stats.json")
print(json.dumps(baselines, indent=2))
