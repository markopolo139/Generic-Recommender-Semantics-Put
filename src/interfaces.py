from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union

class EmbeddingGenerator(ABC):
    """
    Abstract base class for methods that create embeddings.
    """

    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        """
        Trains the model or generates embeddings based on the provided interactions.
        
        Args:
            interactions_df: A DataFrame containing interactions (e.g., user_id, item_id, rating).
        """
        pass

    def train(self, interactions_df: pd.DataFrame, **kwargs) -> None:
        """
        Alias for fit method. Trains the model.
        """
        self.fit(interactions_df, **kwargs)

    @abstractmethod
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Returns the generated embeddings.

        Returns:
            A dictionary mapping entity IDs (users/items) to their embedding vectors.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the model or embeddings to the specified path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the model or embeddings from the specified path.
        """
        pass
