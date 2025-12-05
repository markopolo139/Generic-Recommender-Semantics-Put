import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any

from src.interfaces import EmbeddingGenerator

class LightGCNModel(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3): 
        super().__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index):
        # Initial embeddings (E^0)
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        # Calculate normalization for symmetric adjacency matrix
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Propagation loop
        final_embs = (1 / (self.num_layers + 1)) * x
        
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm)
            final_embs += (1 / (self.num_layers + 1)) * x
            
        user_final_embs, item_final_embs = torch.split(final_embs, [self.num_users, self.num_items])
        
        return user_final_embs, item_final_embs

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class BPRDataset(Dataset):
    def __init__(self, interactions, num_items, user_map, item_col='item_idx', user_col='user_idx'): 
        self.interactions = interactions
        self.num_items = num_items
        self.user_map = user_map
        self.item_col = item_col
        self.user_col = user_col
        
        self.user_pos_items = self.interactions.groupby(user_col)[item_col].apply(set)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        user_idx = interaction[self.user_col]
        pos_item_idx = interaction[self.item_col]
        
        neg_item_idx = random.randint(0, self.num_items - 1)
        while neg_item_idx in self.user_pos_items[user_idx]:
            neg_item_idx = random.randint(0, self.num_items - 1)
            
        return user_idx, pos_item_idx, neg_item_idx

def bpr_loss(users_emb, pos_items_emb, neg_items_emb, users_emb_0, pos_items_emb_0, neg_items_emb_0, lambda_reg=1e-4): 
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    
    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    
    reg_loss = (users_emb_0.norm(2).pow(2) + 
               pos_items_emb_0.norm(2).pow(2) + 
               neg_items_emb_0.norm(2).pow(2)) / 2
               
    return loss + lambda_reg * reg_loss

class LightGCNGenerator(EmbeddingGenerator):
    def __init__(self, embedding_dim=64, num_layers=3, batch_size=4096, learning_rate=1e-3, epochs=10, device='cuda'):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        # Assuming interactions_df has 'user_id' and 'item_id' columns
        user_col = kwargs.get('user_col', 'user_id')
        item_col = kwargs.get('item_col', 'item_id')
        
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()

        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.item_map = {item: i for i, item in enumerate(unique_items)}
        self.inv_user_map = {i: user for user, i in self.user_map.items()}
        self.inv_item_map = {i: item for item, i in self.item_map.items()}

        num_users = len(self.user_map)
        num_items = len(self.item_map)
        
        interactions_df = interactions_df.copy()
        interactions_df['user_idx'] = interactions_df[user_col].map(self.user_map)
        interactions_df['item_idx'] = interactions_df[item_col].map(self.item_map)

        user_indices = torch.LongTensor(interactions_df['user_idx'].values)
        item_indices = torch.LongTensor(interactions_df['item_idx'].values)

        self.edge_index = torch.stack([
            torch.cat([user_indices, item_indices + num_users]),
            torch.cat([item_indices + num_users, user_indices])
        ], dim=0).to(self.device)
        
        self.model = LightGCNModel(num_users, num_items, self.embedding_dim, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_dataset = BPRDataset(interactions_df, num_items, self.user_map)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for user_batch, pos_item_batch, neg_item_batch in progress_bar:
                optimizer.zero_grad()
                
                user_final_embs, item_final_embs = self.model(self.edge_index)
                
                user_embs = user_final_embs[user_batch.to(self.device)]
                pos_item_embs = item_final_embs[pos_item_batch.to(self.device)]
                neg_item_embs = item_final_embs[neg_item_batch.to(self.device)]
                
                user_embs_0 = self.model.user_embedding(user_batch.to(self.device))
                pos_item_embs_0 = self.model.item_embedding(pos_item_batch.to(self.device))
                neg_item_embs_0 = self.model.item_embedding(neg_item_batch.to(self.device))
                
                loss = bpr_loss(user_embs, pos_item_embs, neg_item_embs, 
                                user_embs_0, pos_item_embs_0, neg_item_embs_0)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        if self.model is None or not hasattr(self, 'edge_index'):
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        with torch.no_grad():
            user_final_embs, item_final_embs = self.model(self.edge_index)
            
        embeddings = {}
        user_embs_np = user_final_embs.cpu().numpy()
        item_embs_np = item_final_embs.cpu().numpy()
        
        for i, user_id in self.inv_user_map.items():
            embeddings[user_id] = user_embs_np[i]
            
        for i, item_id in self.inv_item_map.items():
            embeddings[item_id] = item_embs_np[i]
            
        return embeddings

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_map': self.user_map,
            'item_map': self.item_map
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.user_map = checkpoint['user_map']
        self.item_map = checkpoint['item_map']
        self.inv_user_map = {i: u for u, i in self.user_map.items()}
        
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        
        self.model = LightGCNModel(num_users, num_items, self.embedding_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
