import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.features import build_feature_matrix

# Load full student info (contains demographic columns)
info_raw = pd.read_csv('data/raw/studentInfo.csv')
info_raw = info_raw[info_raw['code_module'] == 'BBB'].copy()

# Load at_risk labels and merge
labels = pd.read_csv('data/processed/student_labels.csv')
info = info_raw.merge(labels[['id_student', 'at_risk']], on='id_student', how='inner')

# Load other datasets
vle = pd.read_csv('data/raw/studentVle.csv')
vle = vle[vle['code_module'] == 'BBB']
assess = pd.read_csv('data/raw/studentAssessment.csv')
assess_meta = pd.read_csv('data/raw/assessments.csv')
reg = pd.read_csv('data/raw/studentRegistration.csv')

# Build feature matrix for week 8 (default)
WEEK_NUM = 8
X = build_feature_matrix(info, vle, assess, assess_meta, reg, WEEK_NUM)

# Save
os.makedirs('data/features', exist_ok=True)
out_path = f'data/features/features_week_{WEEK_NUM}.csv'
X.to_csv(out_path, index=False)
print(f"Feature matrix saved to {out_path}")
