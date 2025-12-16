import pandas as pd
import numpy as np
import torch
import os
from src.data_loaders import load_movielens_data, load_steam_data
from src.evaluation import random_split, calculate_metrics, print_metrics
from src.models.matrix_factorization import MatrixFactorizationGenerator

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    movies_df, ratings_df = load_movielens_data()
    reviews_df_steam, items_df_steam = load_steam_data()

    # Preprocessing (similar to notebook)
    movielens_interactions = pd.DataFrame({
        'user_id': 'movielens_user_' + ratings_df['userId'].astype(str),
        'item_id': 'movielens_item_' + ratings_df['movieId'].astype(str),
        'rating': ratings_df['rating']
    })

    # For Steam, explicit ratings are not available in the same way, but we can infer or use 1.0
    # The notebook used 1.0
    reviews_df_steam_filtered = reviews_df_steam.copy()
    steam_interactions = pd.DataFrame({
        'user_id': 'steam_user_' + reviews_df_steam_filtered['user_id'].astype(str),
        'item_id': 'steam_item_' + reviews_df_steam_filtered['app_id'].astype(str),
        'rating': 1.0 
    })

    all_interactions = pd.concat([movielens_interactions, steam_interactions]).drop_duplicates()
    print(f"Total unique interactions: {len(all_interactions)}")

    # --- Data Splitting ---
    print("Splitting data...")
    train_interactions, test_interactions = random_split(all_interactions, test_size=0.2)
    print(f"Training interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")

    # --- Training ---
    print("Initializing model...")
    # Parameters from notebook
    embedding_dim = 32
    batch_size = 4096
    learning_rate = 1e-3
    epochs = 1
    lambda_reg = 1e-5

    generator = MatrixFactorizationGenerator(
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        lambda_reg=lambda_reg,
        device=device
    )

    print("Starting training...")
    generator.fit(train_interactions, user_col='user_id', item_col='item_id', rating_col='rating')

    # --- Saving ---
    os.makedirs('models', exist_ok=True)
    model_path = 'models/matrix_factorization_model.pth'
    generator.save(model_path)
    print(f"Model saved to {model_path}")

    # --- Evaluation ---
    print("Evaluating...")
    # Using generic calculate_metrics
    metrics = calculate_metrics(
        generator, 
        train_interactions, 
        test_interactions, 
        user_col='user_id', 
        item_col='item_id',
        k_list=[10, 20]
    )
    
    print_metrics(metrics)

if __name__ == "__main__":
    main()
