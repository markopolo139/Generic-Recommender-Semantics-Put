import pandas as pd
import numpy as np
from typing import Dict, Any
import itertools

from src.interfaces import EmbeddingGenerator

class CleoraGenerator(EmbeddingGenerator):
    def __init__(self, embedding_dim=128, num_walks=5, columns='complex::reflexive::entity'):
        self.embedding_dim = embedding_dim
        self.num_walks = num_walks
        self.columns = columns
        self.embeddings = {}

    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        try:
            from pycleora import SparseMatrix
        except ImportError:
            raise ImportError("pycleora is not installed. Please install it to use CleoraGenerator.")

        user_col = kwargs.get('user_col', 'user_id')
        item_col = kwargs.get('item_col', 'item_id')

        # Grouping
        user_groups = interactions_df.groupby(user_col)[item_col].apply(list)
        item_groups = interactions_df.groupby(item_col)[user_col].apply(list)
        
        # iterators
        def join_group(group):
            for x in group:
                if isinstance(x, list) and len(x) > 0:
                    yield ' '.join(map(str, x))
                
        iter1 = join_group(user_groups)
        iter2 = join_group(item_groups)
        
        cleora_input = itertools.chain(iter1, iter2)
        
        mat = SparseMatrix.from_iterator(cleora_input, columns=self.columns)
        embeddings_matrix = mat.initialize_deterministically(self.embedding_dim)
        
        for i in range(self.num_walks):
            embeddings_matrix = mat.left_markov_propagate(embeddings_matrix)
            embeddings_matrix /= np.linalg.norm(embeddings_matrix, ord=2, axis=-1, keepdims=True)
            
        self.embeddings = {entity: embedding for entity, embedding in zip(mat.entity_ids, embeddings_matrix)}

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        return self.embeddings

    def save(self, path: str) -> None:
        # Save embeddings as dict
        import torch
        torch.save(self.embeddings, path)

    def load(self, path: str) -> None:
        import torch
        self.embeddings = torch.load(path, weights_only=False)