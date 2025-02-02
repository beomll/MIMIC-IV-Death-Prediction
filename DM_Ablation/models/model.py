import torch
import torch.nn as nn
from transformers import BertModel


class GatedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(GatedTransformerLayer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        # 디버깅: src의 크기와 gate의 입력 크기를 확인
        if src.shape[-1] != self.gate.in_features:
            raise ValueError(f"Input dimension {src.shape[-1]} does not match gate input size {self.gate.in_features}")

        gate_weights = self.sigmoid(self.gate(src))
        transformer_output = self.transformer_layer(src)
        output = transformer_output * gate_weights
        return output


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        bert_model_name = config['bert_model_name']
        num_item_ids = config['num_item_ids']
        num_units = config['num_value_with_units']
        embedding_dim = config['embedding_dim']
        nhead = config.get('nhead', 4)
        num_layers = config.get('num_layers', 2)

        # BERT 모델 초기화
        self.bert = BertModel.from_pretrained(bert_model_name)

        # GTN 초기화
        self.item_id_embedding = nn.Embedding(num_item_ids, embedding_dim, padding_idx=0)
        self.unit_embedding = nn.Embedding(num_units, embedding_dim)

        # d_model 계산: embedding_dim * 2 + value_seq 추가
        d_model = embedding_dim * 2 + 1
        if d_model % nhead != 0:
            d_model += nhead - (d_model % nhead)  # nhead로 나누어떨어지도록 패딩

        self.d_model = d_model  # 저장하여 다른 부분에서도 사용
        self.gtn_layers = nn.ModuleList([
            GatedTransformerLayer(d_model=d_model, nhead=nhead)
            for _ in range(num_layers)
        ])

        # Feature Fusion Layer
        self.feature_fusion_dim = self.bert.config.hidden_size + d_model
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Output Layers
        self.hospital_expire_flag_mlp = nn.Linear(64, 2)  # 병원 내 사망 여부
        self.within_120hr_death_mlp = nn.Linear(64, 2)    # 120시간 안에 사망 여부

    def forward(self, description_tokens, item_id_seq, unit_seq, value_seq):
        # BERT 임베딩
        bert_output = self.bert(**description_tokens)
        bert_emb = bert_output.pooler_output

        # GTN 임베딩
        item_emb = self.item_id_embedding(item_id_seq)  # (batch_size, seq_len, embedding_dim)
        unit_emb = self.unit_embedding(unit_seq)        # (batch_size, seq_len, embedding_dim)

        # value_seq 확장하여 GTN 입력 차원에 맞춤
        value_seq_expanded = value_seq.unsqueeze(-1)    # (batch_size, seq_len, 1)

        # GTN 입력 결합
        combined_seq_emb = torch.cat([item_emb, unit_emb, value_seq_expanded], dim=-1)  # (batch_size, seq_len, embedding_dim*2 + 1)

        # 패딩 추가하여 d_model에 맞춤
        padding_dim = self.d_model - combined_seq_emb.shape[-1]
        if padding_dim > 0:
            padding = torch.zeros(combined_seq_emb.shape[0], combined_seq_emb.shape[1], padding_dim).to(combined_seq_emb.device)
            combined_seq_emb = torch.cat([combined_seq_emb, padding], dim=-1)

        gtn_input = combined_seq_emb.transpose(0, 1)    # (seq_len, batch_size, feature_dim)

        for layer in self.gtn_layers:
            gtn_input = layer(gtn_input)

        gtn_emb = gtn_input.mean(dim=0)  # (batch_size, d_model)

        # BERT 임베딩과 GTN 임베딩 결합
        combined_emb = torch.cat([bert_emb, gtn_emb], dim=-1)  # (batch_size, feature_fusion_dim)
        fused_features = self.feature_fusion(combined_emb)

        # 병원 내 사망 여부 계산
        hospital_logits = self.hospital_expire_flag_mlp(fused_features)

        # 병원 내 사망 여부가 1일 때만 120시간 내 사망 여부 계산
        hospital_pred = torch.argmax(hospital_logits, dim=1)  # 0: 생존, 1: 사망
        within_120_logits = torch.zeros_like(hospital_logits)  # 기본값

        if torch.any(hospital_pred == 1):  # 병원 사망자가 있는 경우
            death_indices = torch.where(hospital_pred == 1)[0]
            death_features = fused_features[death_indices]  # 사망자에 해당하는 feature만 추출
            within_120_logits[death_indices] = self.within_120hr_death_mlp(death_features)

        return hospital_logits, within_120_logits
