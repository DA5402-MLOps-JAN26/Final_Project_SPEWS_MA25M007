import pandas as pd
import os

# Load student info
info = pd.read_csv('data/raw/studentInfo.csv')
info = info[info['code_module'] == 'BBB']  # Filter to module BBB

# Create binary 'at_risk' label: 1 if final_result is Withdrawn or Fail
info['at_risk'] = info['final_result'].apply(
    lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0
)

# Save labels
os.makedirs('data/processed', exist_ok=True)
info[['id_student', 'at_risk', 'final_result']].to_csv(
    'data/processed/student_labels.csv', index=False
)

print("Student labels shape:", info.shape)
print("At-risk rate: {:.2%}".format(info['at_risk'].mean()))
print("Labels saved to data/processed/student_labels.csv")
