import os
import random
import numpy as np
import torch
import scipy.sparse as sp

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Returns the appropriate device (MPS for Mac, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_text_file(filepath):
    """Loads a text file and returns a list of lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_classes(filepath):
    """
    Loads classes.txt.
    Returns:
        id2class: dict {id: class_name}
        class2id: dict {class_name: id}
    """
    lines = load_text_file(filepath)
    id2class = {}
    class2id = {}
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            cid = int(parts[0])
            cname = parts[1].replace('_', ' ') # Remove underscores
            id2class[cid] = cname
            class2id[cname] = cid
    return id2class, class2id

def load_hierarchy(filepath):
    """
    Loads class_hierarchy.txt.
    Returns:
        edges: list of (parent_id, child_id) tuples
    """
    lines = load_text_file(filepath)
    edges = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            pid = int(parts[0])
            cid = int(parts[1])
            edges.append((pid, cid))
    return edges

def load_keywords(filepath):
    """
    Loads class_related_keywords.txt.
    Returns:
        class2keywords: dict {class_name: [keyword1, keyword2, ...]}
    """
    lines = load_text_file(filepath)
    class2keywords = {}
    for line in lines:
        parts = line.split(':')
        if len(parts) >= 2:
            cname = parts[0].strip()
            keywords = [k.strip() for k in parts[1].split(',')]
            class2keywords[cname] = keywords
    return class2keywords

def load_corpus(filepath):
    """
    Loads train_corpus.txt or test_corpus.txt.
    Returns:
        corpus: dict {doc_id: text}
    """
    lines = load_text_file(filepath)
    corpus = {}
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            did = int(parts[0])
            text = parts[1]
            corpus[did] = text
    return corpus

def build_adjacency_matrix(num_classes, edges, add_self_loops=True):
    """
    Builds the normalized adjacency matrix for GCN.
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    # Create adjacency matrix (undirected/symmetric for information flow)
    adj = sp.coo_matrix((np.ones(len(edges)), (
        [e[0] for e in edges],
        [e[1] for e in edges]
    )), shape=(num_classes, num_classes), dtype=np.float32)

    # Make symmetric (Parent <-> Child)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0])

    # Normalization
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    # Convert to sparse tensor
    norm_adj = norm_adj.tocoo()
    indices = torch.from_numpy(
        np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data).float()
    shape = torch.Size(norm_adj.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

def get_hierarchy_relations(edges, num_classes):
    """
    Returns dictionaries for parent-child relations.
    """
    parents = {i: [] for i in range(num_classes)}
    children = {i: [] for i in range(num_classes)}
    
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)
        
    return parents, children

def get_ancestors(class_id, parents_dict):
    """Recursively find all ancestors of a class."""
    ancestors = set()
    queue = [class_id]
    while queue:
        curr = queue.pop(0)
        for p in parents_dict[curr]:
            if p not in ancestors:
                ancestors.add(p)
                queue.append(p)
    return ancestors

def get_siblings(class_id, parents_dict, children_dict):
    """
    Returns a list of sibling class IDs for a given class_id.
    Siblings share at least one parent.
    """
    parents = parents_dict.get(class_id, [])
    if not parents:
        return [] # Root nodes might be considered siblings if they share a common virtual root, but here we assume no parents = no siblings via parents
    
    siblings = set()
    for p in parents:
        children = children_dict.get(p, [])
        for c in children:
            if c != class_id:
                siblings.add(c)
    return list(siblings)

def get_descendants(class_id, children_dict):
    """Recursively find all descendants of a class."""
    descendants = set()
    queue = [class_id]
    while queue:
        curr = queue.pop(0)
        for c in children_dict[curr]:
            if c not in descendants:
                descendants.add(c)
                queue.append(c)
    return descendants
