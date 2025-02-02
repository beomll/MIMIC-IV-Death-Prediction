# dataset.py

from torch.utils.data import Dataset

class ClinicalDataset(Dataset):
    def __init__(self, clinical_data, time_series_data, labels):
        self.clinical_data = clinical_data
        self.time_series_data = time_series_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "clinical_inputs": self.clinical_data[idx],
            "item_ids": self.time_series_data[idx]['item_ids'],
            "values": self.time_series_data[idx]['values']
        }, self.labels[idx]
