import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.interfaces import EmbeddingGenerator

class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Dot product of user and item embeddings
        rating = torch.sum(user_emb * item_emb, dim=1)
        
        return rating

class RatingDataset(Dataset):
    def __init__(self, interactions, user_col='user_idx', item_col='item_idx', rating_col='rating'):
        self.interactions = interactions
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        return (
            interaction[self.user_col],
            interaction[self.item_col],
            interaction[self.rating_col]
        )

class MatrixFactorizationGenerator(EmbeddingGenerator):
    def __init__(self, embedding_dim=64, batch_size=4096, learning_rate=1e-3, epochs=5, lambda_reg=1e-5, device='cuda'):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        user_col = kwargs.get('user_col', 'user_id')
        item_col = kwargs.get('item_col', 'item_id')
        rating_col = kwargs.get('rating_col', 'rating')
        
        # Check if rating column exists, if not create dummy
        if rating_col not in interactions_df.columns:
            interactions_df = interactions_df.copy()
            interactions_df[rating_col] = 1.0
            
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
        
        train_dataset = RatingDataset(interactions_df, rating_col=rating_col)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = MatrixFactorizationModel(num_users, num_items, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for user_batch, item_batch, rating_batch in progress_bar:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)
                rating_batch = rating_batch.float().to(self.device)
                
                optimizer.zero_grad()
                
                predictions = self.model(user_batch, item_batch)
                
                # MSE Loss
                mse_loss = loss_fn(predictions, rating_batch)
                
                # L2 Regularization
                user_emb_reg = self.model.user_embedding(user_batch).norm(2).pow(2)
                item_emb_reg = self.model.item_embedding(item_batch).norm(2).pow(2)
                reg_loss = self.lambda_reg * (user_emb_reg + item_emb_reg) / len(user_batch)
                
                loss = mse_loss + reg_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        embeddings = {}
        
        user_embs = self.model.user_embedding.weight.detach().cpu().numpy()
        item_embs = self.model.item_embedding.weight.detach().cpu().numpy()
        
        for i, user in self.inv_user_map.items():
            embeddings[user] = user_embs[i]
            
        for i, item in self.inv_item_map.items():
            embeddings[item] = item_embs[i]
            
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
        self.inv_item_map = {i: item for item, i in self.item_map.items()}
        
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        
        self.model = MatrixFactorizationModel(num_users, num_items, self.embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
