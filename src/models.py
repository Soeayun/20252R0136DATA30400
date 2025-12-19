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
    GCN-based Label Encoder. (Original)
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


class LabelGAT(nn.Module):
    """
    GAT-based Label Encoder using PyTorch Geometric.
    Uses multi-head attention for learning importance of parent/child relationships.
    """
    def __init__(self, emb_dim, num_layers=2, num_heads=4, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Each head outputs emb_dim // num_heads dimensions
        head_dim = emb_dim // num_heads
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: single head, no concat
                self.layers.append(GATConv(emb_dim, emb_dim, heads=1, concat=False, dropout=dropout))
            else:
                # Intermediate layers: multi-head with concat
                self.layers.append(GATConv(emb_dim, head_dim, heads=num_heads, concat=True, dropout=dropout))
        
        self.edge_index = None  # Will be set from adj matrix
        
    def _adj_to_edge_index(self, adj):
        """Convert sparse adjacency matrix to edge_index format for PyG"""
        if adj.is_sparse:
            indices = adj.coalesce().indices()
        else:
            indices = adj.nonzero(as_tuple=False).t()
        return indices
        
    def forward(self, x, adj):
        # Convert adj to edge_index (only once or if changed)
        if self.edge_index is None or self.edge_index.device != x.device:
            self.edge_index = self._adj_to_edge_index(adj).to(x.device)
        
        for i, layer in enumerate(self.layers):
            identity = x  # Skip connection
            
            x = layer(x, self.edge_index)
            
            # Skip connection (residual)
            x = x + identity
            
            # Activation & Dropout (except last layer)
            if i < len(self.layers) - 1:
                x = F.elu(x)  # ELU is common for GAT
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class TaxoClassModel(nn.Module):
    """
    Dual Encoder Model: Document Encoder + Label GCN/GAT.
    """
    def __init__(self, num_classes, label_emb_init, adj, model_name='bert-base-uncased', 
                 hidden_dim=768, num_gcn_layers=2, dropout=0.5, use_gat=True, num_heads=4):
        super().__init__()
        self.doc_encoder = DocumentEncoder(model_name, hidden_dim)
        
        # Choose between GCN and GAT
        if use_gat:
            print("Initializing TaxoClassModel with GAT...")
            self.label_encoder = LabelGAT(hidden_dim, num_gcn_layers, num_heads, dropout)
        else:
            print("Initializing TaxoClassModel with GCN...")
            self.label_encoder = LabelGCN(hidden_dim, num_gcn_layers, dropout)
        
        # Label Embeddings (Learnable)
        # Initialize with pre-computed BERT embeddings of class names
        self.label_embeddings = nn.Parameter(label_emb_init.clone())
        
        # Adjacency Matrix (Fixed)
        self.register_buffer('adj', adj)
        
    def forward(self, input_ids, attention_mask):
        # 1. Encode Documents
        doc_emb = self.doc_encoder(input_ids, attention_mask)
        
        # 2. Encode Labels (GCN or GAT)
        label_emb = self.label_encoder(self.label_embeddings, self.adj)
        
        # 3. Compute Logits (Dot Product)
        # doc_emb: (batch, dim)
        # label_emb: (num_classes, dim)
        # logits: (batch, num_classes)
        logits = torch.matmul(doc_emb, label_emb.T)
        
        return logits

