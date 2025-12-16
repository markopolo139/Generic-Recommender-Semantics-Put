import pandas as pd
import torch
import os
from src.data_loaders import load_movielens_data, load_steam_data
from src.evaluation import random_split, calculate_metrics, print_metrics
from src.models.lightgcn import LightGCNGenerator

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    movies_df, ratings_df = load_movielens_data()
    reviews_df_steam, items_df_steam = load_steam_data()

    # Preprocessing
    # Filter high ratings for MovieLens
    ratings_df = ratings_df[ratings_df['rating'] >= 4.0]
    
    movielens_interactions = pd.DataFrame({
        'user_id': 'movielens_user_' + ratings_df['userId'].astype(str),
        'item_id': 'movielens_item_' + ratings_df['movieId'].astype(str)
    })

    steam_interactions = pd.DataFrame({
        'user_id': 'steam_user_' + reviews_df_steam['user_id'].astype(str),
        'item_id': 'steam_item_' + reviews_df_steam['app_id'].astype(str)
    })

    all_interactions = pd.concat([movielens_interactions, steam_interactions]).drop_duplicates()
    print(f"Total unique interactions: {len(all_interactions)}")

    # --- Data Splitting ---
    print("Splitting data...")
    # Using random split as per general practice if not specified otherwise
    train_interactions, test_interactions = random_split(all_interactions, test_size=0.2)
    print(f"Training interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")

    # --- Training ---
    print("Initializing LightGCN model...")
    # Parameters (inferred or default)
    embedding_dim = 64
    num_layers = 3
    batch_size = 4096
    learning_rate = 1e-3
    epochs = 10 

    generator = LightGCNGenerator(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device
    )

    print("Starting training...")
    generator.fit(train_interactions, user_col='user_id', item_col='item_id')

    # --- Saving ---
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lightgcn_model.pth'
    generator.save(model_path)
    print(f"Model saved to {model_path}")

    # --- Evaluation ---
    print("Evaluating...")
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
