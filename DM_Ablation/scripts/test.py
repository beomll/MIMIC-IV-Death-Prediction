import os
from os import path
import sys
import time
sys.path.append(path.abspath('..'))
from datetime import datetime
import torch
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from models.model import Model


class MimicDataset(Dataset):
    def __init__(self, data_file):
        try:
            self.data = torch.load(data_file, weights_only=True)
        except (RuntimeError, pickle.UnpicklingError):
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    input_ids = [sample['input_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]
    token_type_ids = [sample['token_type_ids'] for sample in batch]
    item_id_seq = [sample['item_id_seq'] for sample in batch]
    unit_seq = [sample['unit_seq'] for sample in batch]
    value_seq = [sample['value_seq'] for sample in batch]
    hospital_labels = [sample['hospital_expire_flag'] for sample in batch]
    within_120_labels = [sample['within_120hr_death'] for sample in batch]

    max_len_input = max(len(ids) for ids in input_ids)
    max_len_item = max(len(seq) for seq in item_id_seq)

    input_ids = torch.tensor([ids + [0] * (max_len_input - len(ids)) for ids in input_ids], dtype=torch.long)
    attention_mask = torch.tensor([mask + [0] * (max_len_input - len(mask)) for mask in attention_mask],
                                  dtype=torch.long)
    token_type_ids = torch.tensor([ids + [0] * (max_len_input - len(ids)) for ids in token_type_ids], dtype=torch.long)
    item_id_seq = torch.tensor([seq + [0] * (max_len_item - len(seq)) for seq in item_id_seq], dtype=torch.long)
    unit_seq = torch.tensor([seq + [0] * (max_len_item - len(seq)) for seq in unit_seq], dtype=torch.long)
    value_seq = torch.tensor([seq + [0.0] * (max_len_item - len(seq)) for seq in value_seq], dtype=torch.float)

    hospital_labels = torch.tensor(hospital_labels, dtype=torch.long)
    within_120_labels = torch.tensor(within_120_labels, dtype=torch.long)

    description_tokens = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
    }

    return description_tokens, item_id_seq, unit_seq, value_seq, hospital_labels, within_120_labels


def evaluate_test(model, dataloader, device):
    model.eval()
    all_within_120_preds, all_within_120_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            description_tokens, item_id_seq, unit_seq, value_seq, hospital_labels, within_120_labels = batch
            for key in description_tokens:
                description_tokens[key] = description_tokens[key].to(device)
            item_id_seq = item_id_seq.to(device)
            unit_seq = unit_seq.to(device)
            value_seq = value_seq.to(device)
            hospital_labels = hospital_labels.to(device)
            within_120_labels = within_120_labels.to(device)

            _, within_120_logits = model(description_tokens, item_id_seq, unit_seq, value_seq)

            within_120_preds = torch.argmax(within_120_logits, dim=1)
            all_within_120_preds.extend(within_120_preds[hospital_labels == 1].cpu().numpy())
            all_within_120_labels.extend(within_120_labels[hospital_labels == 1].cpu().numpy())

    within_120_f1 = f1_score(all_within_120_labels, all_within_120_preds, zero_division=0)
    within_120_accuracy = accuracy_score(all_within_120_labels, all_within_120_preds)
    within_120_precision = precision_score(all_within_120_labels, all_within_120_preds, zero_division=0)
    within_120_recall = recall_score(all_within_120_labels, all_within_120_preds, zero_division=0)
    within_120_auc = roc_auc_score(all_within_120_labels, all_within_120_preds)
    cm = confusion_matrix(all_within_120_labels, all_within_120_preds)

    return {
        'f1': within_120_f1,
        'accuracy': within_120_accuracy,
        'precision': within_120_precision,
        'recall': within_120_recall,
        'auc': within_120_auc,
        'confusion_matrix': cm,
    }


def plot_confusion_matrix(cm, class_names):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()


def test():
    with open('../config/hyperparameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    model_params = config['model_params']
    logging_config = config['logging']

    # 데이터 로드
    test_dataset = MimicDataset(os.path.join(data_config['data_dir'], data_config['processed_test_file']))
    test_loader = DataLoader(test_dataset, batch_size=logging_config.get('batch_size', 32), shuffle=False,
                             collate_fn=collate_fn)

    # 모델 초기화 및 로드
    model = Model(model_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Best 모델 경로
    best_model_path = os.path.join(logging_config['log_dir'], '20241201-131233', 'best_model_fold_4.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 테스트 평가
    test_metrics = evaluate_test(model, test_loader, device)

    print("\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")

    # Confusion Matrix 출력
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names=['No', 'Yes'])


if __name__ == '__main__':
    test()
