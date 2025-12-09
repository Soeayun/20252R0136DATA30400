import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DocumentEncoder(nn.Module):
    """
    BERT-based Document Encoder.
    """
    def __init__(self, model_name='microsoft/deberta-v3-base', hidden_dim=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        # DeBERTa v3 uses the first token as [CLS] equivalent
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_emb)

class LabelGCN(nn.Module):
    """
    GCN-based Label Encoder.
    """
    def __init__(self, emb_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(emb_dim, emb_dim))
        self.dropout = dropout

    def forward(self, x, adj):
        # x: (num_classes, emb_dim)
        # adj: (num_classes, num_classes) sparse tensor
        
        for i, layer in enumerate(self.layers):
            identity = x  # Save input for skip connection
            
            # Message Passing: AX
            x = torch.sparse.mm(adj, x)
            # Linear Transform: XW
            x = layer(x)
            
            # Skip Connection (Residual)
            x = x + identity
            
            # Activation & Dropout (except last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class TaxoClassModel(nn.Module):
    """
    Dual Encoder Model: Document Encoder + Label GCN.
    """
    def __init__(self, num_classes, label_emb_init, adj, model_name='bert-base-uncased', hidden_dim=768, num_gcn_layers=2, dropout=0.5):
        super().__init__()
        self.doc_encoder = DocumentEncoder(model_name, hidden_dim)
        self.label_gcn = LabelGCN(hidden_dim, num_gcn_layers, dropout)
        
        # Label Embeddings (Learnable)
        # Initialize with pre-computed BERT embeddings of class names
        self.label_embeddings = nn.Parameter(label_emb_init.clone())
        
        # Adjacency Matrix (Fixed)
        self.register_buffer('adj', adj)
        
    def forward(self, input_ids, attention_mask):
        # 1. Encode Documents
        doc_emb = self.doc_encoder(input_ids, attention_mask)
        
        # 2. Encode Labels (GCN)
        label_emb = self.label_gcn(self.label_embeddings, self.adj)
        
        # 3. Compute Logits (Dot Product)
        # doc_emb: (batch, dim)
        # label_emb: (num_classes, dim)
        # logits: (batch, num_classes)
        logits = torch.matmul(doc_emb, label_emb.T)
        
        return logits
