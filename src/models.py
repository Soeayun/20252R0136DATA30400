import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import geoopt

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

class LabelHGCN(nn.Module):
    """
    Hyperbolic GCN for Label Encoder using Poincaré Ball.
    Better for hierarchical structure than Euclidean GCN.
    """
    def __init__(self, emb_dim, num_layers=2, dropout=0.5, c=1.0):
        super().__init__()
        
        # Poincaré ball manifold with learnable curvature
        self.manifold = geoopt.PoincareBall(c=c)
        self.c = nn.Parameter(torch.tensor([c]))  # Learnable curvature
        
        # Hyperbolic layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Linear transformation in tangent space
            self.layers.append(nn.Linear(emb_dim, emb_dim))
        
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, adj):
        """
        Args:
            x: (num_classes, emb_dim) - Node features in Euclidean space
            adj: (num_classes, num_classes) - Adjacency matrix (sparse)
        
        Returns:
            x: (num_classes, emb_dim) - Node embeddings in Poincaré ball
        """
        # Project initial embeddings to Poincaré ball
        x = self.manifold.expmap0(x, c=self.c)
        
        for i, layer in enumerate(self.layers):
            # 1. Project to tangent space at origin
            x_tan = self.manifold.logmap0(x, c=self.c)
            
            # 2. Graph convolution in tangent space
            # Message passing: A @ X
            x_tan = torch.sparse.mm(adj, x_tan)
            
            # 3. Linear transformation
            x_tan = layer(x_tan)
            
            # 4. Activation & Dropout (except last layer)
            if i < self.num_layers - 1:
                x_tan = F.relu(x_tan)
                x_tan = F.dropout(x_tan, p=self.dropout, training=self.training)
            
            # 5. Project back to Poincaré ball
            x = self.manifold.expmap0(x_tan, c=self.c)
            
            # 6. Ensure points stay in the ball
            x = self.manifold.projx(x, c=self.c)
        
        # Project back to Euclidean for compatibility
        x = self.manifold.logmap0(x, c=self.c)
        
        return x

class TaxoClassModel(nn.Module):
    """
    Dual Encoder Model: Document Encoder + Label HGCN.
    Uses Hyperbolic geometry for better hierarchical structure modeling.
    """
    def __init__(self, num_classes, label_emb_init, adj, model_name='bert-base-uncased', 
                 hidden_dim=768, num_gcn_layers=2, dropout=0.5, hyperbolic_c=1.0):
        super().__init__()
        self.doc_encoder = DocumentEncoder(model_name, hidden_dim)
        
        # Use HGCN instead of GCN for hierarchical structure
        self.label_gcn = LabelHGCN(hidden_dim, num_gcn_layers, dropout, c=hyperbolic_c)
        
        # Label Embeddings (Learnable)
        # Initialize with pre-computed BERT embeddings of class names
        self.label_embeddings = nn.Parameter(label_emb_init.clone())
        
        # Adjacency Matrix (Fixed)
        self.register_buffer('adj', adj)
        
    def forward(self, input_ids, attention_mask):
        # 1. Encode Documents
        doc_emb = self.doc_encoder(input_ids, attention_mask)
        
        # 2. Encode Labels (HGCN in Poincaré ball)
        label_emb = self.label_gcn(self.label_embeddings, self.adj)
        
        # 3. Compute Logits (Dot Product)
        # doc_emb: (batch, dim)
        # label_emb: (num_classes, dim)
        # logits: (batch, num_classes)
        logits = torch.matmul(doc_emb, label_emb.T)
        
        return logits
