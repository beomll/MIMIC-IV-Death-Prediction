import os
import pickle
import pandas as pd
from transformers import BertTokenizer
import torch
import yaml
from tqdm import tqdm


def preprocess_data(input_file, output_file, item_id_to_idx, value_with_unit_to_idx, max_seq_length=500, tokenizer_name='bert-base-uncased'):
    # 데이터 로드
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    processed_data = []

    for sample in tqdm(data):
        hadm_id = sample['hadm_id']
        description = sample['description']
        items = sample['items']
        hospital_expire_flag = sample['hospital_expire_flag']
        within_120hr_death = sample['within_120hr_death']

        # 음수 값 처리
        hospital_expire_flag = max(0, hospital_expire_flag)
        within_120hr_death = max(0, within_120hr_death)

        # description 처리
        description_tokens = tokenizer(description, padding='max_length', truncation=True, max_length=512)
        input_ids = description_tokens['input_ids']
        attention_mask = description_tokens['attention_mask']
        token_type_ids = description_tokens.get('token_type_ids', [0] * len(input_ids))

        # items 처리 및 정렬
        item_sequences = []
        for item in items:
            item_id = item['item_id']
            item_idx = item_id_to_idx.get(item_id, len(item_id_to_idx))
            item_id_to_idx[item_id] = item_idx  # 새로운 item_id 추가

            unit = item['unit']
            measurements = item['measurements']

            for measurement in measurements:
                start_time = pd.to_datetime(measurement['start_time'])  # 시간 정보
                value = measurement['value']
                value = round(value, 2)

                value_with_unit = f"{value} {unit}"
                value_with_unit_idx = value_with_unit_to_idx.get(value_with_unit, len(value_with_unit_to_idx))
                value_with_unit_to_idx[value_with_unit] = value_with_unit_idx  # 새로운 value_with_unit 추가

                # 정렬을 위한 데이터 추가
                item_sequences.append({
                    'start_time': start_time,
                    'item_id': item_idx,
                    'value_with_unit_idx': value_with_unit_idx,
                })

        # start_time 기준 정렬
        item_sequences.sort(key=lambda x: x['start_time'])

        # 시계열 데이터 생성
        seq_len = min(len(item_sequences), max_seq_length)
        item_id_seq = torch.zeros(max_seq_length, dtype=torch.long)
        value_seq = torch.zeros(max_seq_length, dtype=torch.long)

        for i in range(seq_len):
            item_id_seq[i] = item_sequences[i]['item_id']
            value_seq[i] = item_sequences[i]['value_with_unit_idx']

        # processed_sample 생성
        processed_sample = {
            'hadm_id': hadm_id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'item_id_seq': item_id_seq.tolist(),
            'value_seq': value_seq.tolist(),
            'hospital_expire_flag': hospital_expire_flag,
            'within_120hr_death': within_120hr_death,
        }

        processed_data.append(processed_sample)

    # 데이터 저장
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    return item_id_to_idx, value_with_unit_to_idx


if __name__ == '__main__':
    # 하이퍼파라미터 로드
    with open('../config/hyperparameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    model_config = config['models']
    model_params = config['model_params']

    data_dir = data_config['data_dir']
    data_file = data_config['data_file']  # 통합된 데이터
    test_file = data_config['test_file']
    processed_data_file = data_config['processed_data_file']
    processed_test_file = data_config['processed_test_file']

    max_seq_length = model_params.get('max_seq_length', 500)
    tokenizer_name = model_config['bert_model_name']

    # 전체 데이터에서 item_id와 value_with_unit 수집
    item_id_to_idx = {}
    value_with_unit_to_idx = {}

    for input_file_name in tqdm([data_file, test_file]):
        input_file = os.path.join(data_dir, input_file_name)
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
            for sample in data:
                items = sample['items']
                for item in items:
                    item_id = item['item_id']
                    item_id_to_idx.setdefault(item_id, len(item_id_to_idx))

                    unit = item['unit']
                    measurements = item['measurements']
                    for measurement in measurements:
                        value = round(measurement['value'], 2)
                        value_with_unit = f"{value} {unit}"
                        value_with_unit_to_idx.setdefault(value_with_unit, len(value_with_unit_to_idx))

    # 데이터셋 전처리
    for input_file_name, output_file_name in [
        (data_file, processed_data_file),
        (test_file, processed_test_file)
    ]:
        input_file = os.path.join(data_dir, input_file_name)
        output_file = os.path.join(data_dir, output_file_name)
        item_id_to_idx, value_with_unit_to_idx = preprocess_data(
            input_file,
            output_file,
            item_id_to_idx,
            value_with_unit_to_idx,
            max_seq_length=max_seq_length,
            tokenizer_name=tokenizer_name
        )

    # 모델 파라미터 업데이트
    model_params['num_item_ids'] = len(item_id_to_idx)
    model_params['num_value_with_units'] = len(value_with_unit_to_idx)

    # 업데이트된 모델 파라미터 저장
    config['model_params'] = model_params
    with open('../config/hyperparameters.yaml', 'w') as f:
        yaml.dump(config, f)
