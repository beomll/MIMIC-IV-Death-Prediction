import pandas as pd
from tqdm import tqdm
import pickle
import json

# CSV 파일 불러오기
admissions_df = pd.read_csv('data/test/admissions_test.csv')
patients_df = pd.read_csv('data/test/patients_test.csv')
icu_df = pd.read_csv('data/test/icustays_test.csv')

# ICU 데이터에서 중복된 hadm_id의 los 합산
icu_los_sum = icu_df.groupby('hadm_id', as_index=False)['los'].sum()
icu_los_sum['los'] = icu_los_sum['los'].round(2)  # 소수점 둘째 자리로 반올림

# 'subject_id'를 기준으로 admissions_df와 patients_df 병합
merged_df = pd.merge(admissions_df, patients_df, on='subject_id', how='inner')

# ICU 데이터의 합산된 los를 포함하여 병합
merged_df = pd.merge(merged_df, icu_los_sum, on='hadm_id', how='inner')

# JSON-like 데이터 생성
patient_data = []
for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
    entry = {
        "hadm_id": row['hadm_id'],
        "description": f"The patient's identification number is {row['hadm_id']}. The race is {row['race']}. The gender is {row['gender']}. The age is {row['anchor_age']}. The total ICU length of stay is {row['los']} days."
    }
    patient_data.append(entry)

# 결과를 .pkl 파일로 저장
with open('data/patient_data_test.pkl', 'wb') as file:
    pickle.dump(patient_data, file)

# 결과를 .json 파일로 저장
with open('data/patient_data_test.json', 'w', encoding='utf-8') as file:
    json.dump(patient_data, file, ensure_ascii=False, indent=4)