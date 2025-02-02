import pandas as pd
from tqdm import tqdm

# 데이터셋 불러오기
file_path = 'data/test/admissions_test.csv'
data = pd.read_csv(file_path)

# admittime과 deathtime을 datetime 형식으로 변환
data['admittime'] = pd.to_datetime(data['admittime'], errors='coerce')
data['deathtime'] = pd.to_datetime(data['deathtime'], errors='coerce')

# 새로운 열을 기본 값으로 초기화
data['within_120hr_death'] = -1

# 조건에 따라 within_120hr_death 값을 계산
for i in tqdm(range(len(data)), desc="within_120hr_death 처리 중"):
    # hospital_expire_flag가 1인 경우 (사망)
    if data.loc[i, 'hospital_expire_flag'] == 1:
        # admittime과 deathtime이 모두 유효한 경우
        if pd.notna(data.loc[i, 'admittime']) and pd.notna(data.loc[i, 'deathtime']):
            # admittime과 deathtime의 시간 차이를 계산하여 120시간 이내인지 확인
            time_diff = (data.loc[i, 'deathtime'] - data.loc[i, 'admittime']).total_seconds() / 3600
            data.loc[i, 'within_120hr_death'] = 1 if time_diff <= 120 else 0
    else:
        # 생존의 경우 -1로 설정
        data.loc[i, 'within_120hr_death'] = -1

# 새로운 CSV 파일로 저장
output_path = 'data/admissions_test_modified.csv'
data.to_csv(output_path, index=False, encoding='utf-8')
