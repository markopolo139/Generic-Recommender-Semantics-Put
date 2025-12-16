import pandas as pd
import numpy as np
import os
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loaders import load_movielens_data, load_steam_data

def main():
    try:
        from pycleora import SparseMatrix
    except ImportError:
        print("pycleora is not installed. Please install it to run this script.")
        return

    print("Loading data...")
    movies_df, ratings_df = load_movielens_data()
    reviews_df_steam, items_df_steam = load_steam_data()

    # --- Preprocessing & Graph Construction with Genres ---
    print("Preparing interactions...")
    
    # 1. User-Item Interactions
    ratings_df_filtered = ratings_df[ratings_df['rating'] >= 4.0].copy()
    movielens_interactions = pd.DataFrame({
        'user_id': 'movielens_user_' + ratings_df_filtered['userId'].astype(str),
        'item_id': 'movielens_item_' + ratings_df_filtered['movieId'].astype(str)
    })

    steam_interactions = pd.DataFrame({
        'user_id': 'steam_user_' + reviews_df_steam['user_id'].astype(str),
        'item_id': 'steam_item_' + reviews_df_steam['app_id'].astype(str)
    })

    all_interactions = pd.concat([movielens_interactions, steam_interactions]).drop_duplicates()
    
    user_items = all_interactions.groupby('user_id')['item_id'].apply(list).values
    item_users = all_interactions.groupby('item_id')['user_id'].apply(list).values

    # 2. Item-Genre Interactions
    print("Preparing genre interactions...")
    
    # MovieLens Genres
    movielens_genres = movies_df.copy()
    movielens_genres['genres'] = movielens_genres['genres'].str.split('|')
    movielens_genres = movielens_genres.explode('genres')
    movielens_genres = movielens_genres[movielens_genres['genres'] != '(no genres listed)']
    movielens_genres['item_id'] = 'movielens_item_' + movielens_genres['movieId'].astype(str)
    movielens_genres['genre_id'] = 'genre_' + movielens_genres['genres'].str.lower().str.replace(' ', '_').str.replace('-', '_')
    movielens_item_genre_interactions = movielens_genres[['item_id', 'genre_id']].drop_duplicates()

    # Steam Genres
    steam_genres = items_df_steam.copy()
    steam_genres = steam_genres.dropna(subset=['genres'])
    # Ensure genres is list
    # steam_genres['genres'] is already list from load_steam_data?? 
    # load_steam_data uses read_json. If formatted correctly, it is list.
    # But let's check or be safe. 
    # In notebook: steam_genres['genres'].apply(lambda x: x if isinstance(x, list) else [])
    
    steam_genres['genres'] = steam_genres['genres'].apply(lambda x: x if isinstance(x, list) else [])
    steam_genres = steam_genres.explode('genres')
    steam_genres['item_id'] = 'steam_item_' + steam_genres['app_id'].astype(str)
    steam_genres['genre_id'] = 'genre_' + steam_genres['genres'].str.lower().str.replace(' ', '_').str.replace('-', '_')
    steam_item_genre_interactions = steam_genres[['item_id', 'genre_id']].drop_duplicates()

    all_item_genre_interactions = pd.concat([movielens_item_genre_interactions, steam_item_genre_interactions])

    item_genres = all_item_genre_interactions.groupby('item_id')['genre_id'].apply(list).values
    genre_items = all_item_genre_interactions.groupby('genre_id')['item_id'].apply(list).values

    # --- Training ---
    print("Training Cleora...")
    
    cleora_input_with_genres = itertools.chain(
        map(lambda x: ' '.join(x), user_items), 
        map(lambda x: ' '.join(x), item_users), 
        map(lambda x: ' '.join(x), item_genres), 
        map(lambda x: ' '.join(x), genre_items)
    )

    mat = SparseMatrix.from_iterator(cleora_input_with_genres, columns='complex::reflexive::entity')
    embeddings_matrix = mat.initialize_deterministically(128)

    NUM_WALKS = 5
    for i in range(NUM_WALKS):
        embeddings_matrix = mat.left_markov_propagate(embeddings_matrix)
        embeddings_matrix /= np.linalg.norm(embeddings_matrix, ord=2, axis=-1, keepdims=True)

    embeddings = {entity: embedding for entity, embedding in zip(mat.entity_ids, embeddings_matrix)}
    print(f"Cleora training complete. Loaded {len(embeddings)} embeddings.")

    # --- Saving ---
    os.makedirs('models', exist_ok=True)
    # Saving as pickle or similar might be best for dict of numpy arrays
    import pickle
    with open('models/cleora_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings saved to models/cleora_embeddings.pkl")

    # --- Simple Check ---
    sample_user_id = 'steam_user_LydiaMorley'
    if sample_user_id in embeddings:
        print(f"Embeddings found for {sample_user_id}")
    else:
        print(f"User {sample_user_id} not found in embeddings (might be filtered out).")

if __name__ == "__main__":
    main()
