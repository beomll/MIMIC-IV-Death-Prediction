import os
from os import path
import sys
import time
sys.path.append(path.abspath('..'))
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.model1 import Model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


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


def evaluate(model, dataloader, device):
    model.eval()
    all_within_120_preds, all_within_120_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
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

    return {
        'f1': within_120_f1,
        'accuracy': within_120_accuracy,
        'precision': within_120_precision,
        'recall': within_120_recall,
        'auc': within_120_auc,
    }


def grid_search(hyperparameter_grid):
    with open('../config/hyperparameters.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    model_params = config['model_params']
    logging_config = config['logging']
    training_config = config['training']

    dataset = MimicDataset(os.path.join(data_config['data_dir'], data_config['processed_data_file']))
    kfold = KFold(n_splits=5, shuffle=True, random_state=logging_config.get('random_seed', 42))

    best_f1 = 0.0
    best_params = None
    best_model_path = None

    # Hyperparameter combinations
    param_combinations = list(itertools.product(*hyperparameter_grid.values()))

    for params in param_combinations:
        param_dict = dict(zip(hyperparameter_grid.keys(), params))
        print(f"Testing hyperparameters: {param_dict}")

        # Update model and training configuration
        model_params.update(param_dict)
        learning_rate = float(param_dict.get('learning_rate', training_config['learning_rate']))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\nFold {fold + 1}/5")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=training_config['batch_size'], shuffle=True,
                                      collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=training_config['batch_size'], shuffle=False,
                                    collate_fn=collate_fn)

            model = Model(model_params)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            criterion_hospital = nn.CrossEntropyLoss()
            criterion_within_120 = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=training_config.get('weight_decay', 0.01)
            )

            for epoch in range(training_config['num_epochs']):
                model.train()
                for batch in tqdm(train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1} Training"):
                    description_tokens, item_id_seq, unit_seq, value_seq, hospital_labels, within_120_labels = batch

                    for key in description_tokens:
                        description_tokens[key] = description_tokens[key].to(device)
                    item_id_seq = item_id_seq.to(device)
                    unit_seq = unit_seq.to(device)
                    value_seq = value_seq.to(device)
                    hospital_labels = hospital_labels.to(device)
                    within_120_labels = within_120_labels.to(device)

                    optimizer.zero_grad()

                    hospital_logits, within_120_logits = model(description_tokens, item_id_seq, unit_seq, value_seq)

                    loss_hospital = criterion_hospital(hospital_logits, hospital_labels)
                    loss_within_120 = criterion_within_120(
                        within_120_logits[hospital_labels == 1], within_120_labels[hospital_labels == 1]
                    )
                    loss = loss_hospital + loss_within_120

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
                    optimizer.step()

                val_metrics = evaluate(model, val_loader, device)
                print(f"Validation Results: {val_metrics}")

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    best_params = param_dict
                    best_model_path = f"../logs/best_model_{fold + 1}.pth"
                    torch.save(model.state_dict(), best_model_path)

    print("\nGrid Search Complete!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Model Saved at: {best_model_path}")



if __name__ == '__main__':
    hyperparameter_grid = {
        'dropout_rate': [0.2],
        'nhead': [4],
        'num_layers': [2],
        'learning_rate': [7e-5, 5e-6]
    }

    grid_search(hyperparameter_grid)
