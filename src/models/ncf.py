import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from typing import Dict, Any, List

from src.interfaces import EmbeddingGenerator

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim_gmf, embedding_dim_mlp, mlp_layers):
        super().__init__()
        
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim_gmf)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim_gmf)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim_mlp)
        
        # MLP layers
        self.mlp = nn.Sequential()
        input_size = 2 * embedding_dim_mlp
        for i, layer_size in enumerate(mlp_layers):
            self.mlp.add_module(f"linear_{i}", nn.Linear(input_size, layer_size))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            input_size = layer_size
            
        # Final prediction layer
        self.predict_layer = nn.Linear(embedding_dim_gmf + mlp_layers[-1], 1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user_indices, item_indices):
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_indices)
        item_emb_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_indices)
        item_emb_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP parts
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        
        # Final prediction
        prediction = self.predict_layer(concat)
        
        return prediction.squeeze()

class NCFDataset(Dataset):
    def __init__(self, interactions, num_items, user_map, num_neg_samples=4, user_col='user_idx', item_col='item_idx'):
        self.interactions = interactions
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.user_col = user_col
        self.item_col = item_col
        
        self.user_pos_items = self.interactions.groupby(user_col)[item_col].apply(set)
        
        self.users = []
        self.items = []
        self.labels = []
        
        # Efficient sampling could be improved but following notebook logic
        for _, row in self.interactions.iterrows():
            user_idx = row[user_col]
            pos_item_idx = row[item_col]
            
            # Add positive sample
            self.users.append(user_idx)
            self.items.append(pos_item_idx)
            self.labels.append(1.0)
            
            # Add negative samples
            for _ in range(self.num_neg_samples):
                neg_item_idx = random.randint(0, self.num_items - 1)
                while neg_item_idx in self.user_pos_items[user_idx]:
                    neg_item_idx = random.randint(0, self.num_items - 1)
                
                self.users.append(user_idx)
                self.items.append(neg_item_idx)
                self.labels.append(0.0)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class NCFGenerator(EmbeddingGenerator):
    def __init__(self, embedding_dim_gmf=32, embedding_dim_mlp=32, mlp_layers=[64, 32, 16], 
                 batch_size=1024, learning_rate=1e-3, epochs=5, num_neg_samples=4, device='cuda'):
        self.embedding_dim_gmf = embedding_dim_gmf
        self.embedding_dim_mlp = embedding_dim_mlp
        self.mlp_layers = mlp_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_neg_samples = num_neg_samples
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
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
        
        train_dataset = NCFDataset(interactions_df, num_items, self.user_map, num_neg_samples=self.num_neg_samples)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = NCFModel(num_users, num_items, self.embedding_dim_gmf, self.embedding_dim_mlp, self.mlp_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for user_batch, item_batch, label_batch in progress_bar:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)
                label_batch = label_batch.float().to(self.device)
                
                optimizer.zero_grad()
                
                predictions = self.model(user_batch, item_batch)
                
                loss = loss_fn(predictions, label_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self.model.eval()
        embeddings = {}
        
        with torch.no_grad():
            user_gmf = self.model.user_embedding_gmf.weight
            user_mlp = self.model.user_embedding_mlp.weight
            item_gmf = self.model.item_embedding_gmf.weight
            item_mlp = self.model.item_embedding_mlp.weight
            
            # Concatenate to get full representation
            user_embs = torch.cat([user_gmf, user_mlp], dim=-1).cpu().numpy()
            item_embs = torch.cat([item_gmf, item_mlp], dim=-1).cpu().numpy()
        
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
        
        self.model = NCFModel(num_users, num_items, self.embedding_dim_gmf, self.embedding_dim_mlp, self.mlp_layers).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def legacy_load(self, path: str, interactions_df: pd.DataFrame, user_col: str = 'user_id', item_col: str = 'item_id') -> None:
        """
        Loads a legacy checkpoint that only contains the model state dict (no maps).
        Reconstructs maps from interactions_df.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct maps
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()

        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.item_map = {item: i for i, item in enumerate(unique_items)}
        self.inv_user_map = {i: user for user, i in self.user_map.items()}
        self.inv_item_map = {i: item for item, i in self.item_map.items()}

        num_users = len(self.user_map)
        num_items = len(self.item_map)
        
        self.model = NCFModel(num_users, num_items, self.embedding_dim_gmf, self.embedding_dim_mlp, self.mlp_layers).to(self.device)
        self.model.load_state_dict(checkpoint)
