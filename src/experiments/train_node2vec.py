import pandas as pd
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import Node2Vec
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_loaders import (
    load_movielens_data, 
    load_steam_data, 
    transform_movielenses_to_graph, 
    transform_steam_to_graph, 
    merge_graphs, 
    create_pyg_graph
)

def evaluate_node2vec(embeddings, test_interactions, train_interactions, node_to_int_id, user_map, item_map, device, k=20):
    user_int_indices = [node_to_int_id[uid] for uid in user_map.keys() if uid in node_to_int_id]
    item_int_indices = [node_to_int_id[iid] for iid in item_map.keys() if iid in node_to_int_id]

    user_idx_to_emb_idx = {user_map[uid]: i for i, uid in enumerate(user_map.keys()) if uid in node_to_int_id}
    item_idx_to_emb_idx = {item_map[iid]: i for i, iid in enumerate(item_map.keys()) if iid in node_to_int_id}

    user_embs = embeddings[user_int_indices].to(device)
    item_embs = embeddings[item_int_indices].to(device)

    test_user_indices = test_interactions['user_idx'].unique()
    train_ground_truth = train_interactions.groupby('user_idx')['item_idx'].apply(list).to_dict()
    test_ground_truth = test_interactions.groupby('user_idx')['item_idx'].apply(list).to_dict()

    recalls, precisions, f1_scores = [], [], []

    for user_idx in tqdm(test_user_indices, desc='Evaluating'):
        if user_idx not in user_idx_to_emb_idx: continue

        emb_idx = user_idx_to_emb_idx[user_idx]
        user_emb = user_embs[emb_idx]

        # Dot product scores
        scores = torch.matmul(user_emb, item_embs.T)

        excluded_items = train_ground_truth.get(user_idx, [])
        if excluded_items:
            excluded_emb_indices = [item_idx_to_emb_idx[i] for i in excluded_items if i in item_idx_to_emb_idx]
            if excluded_emb_indices:
                scores[excluded_emb_indices] = -np.inf

        _, top_k_indices = torch.topk(scores, k=k)
        top_k_item_indices = [list(item_idx_to_emb_idx.keys())[i] for i in top_k_indices.cpu().numpy()]

        ground_truth_items = test_ground_truth.get(user_idx, [])
        if not ground_truth_items: continue

        hits = np.isin(top_k_item_indices, ground_truth_items)
        num_hits = np.sum(hits)

        recall = num_hits / len(ground_truth_items)
        precision = num_hits / k
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    avg_recall = np.mean(recalls) if recalls else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    print(f'Recall@{k}: {avg_recall:.4f}')
    print(f'Precision@{k}: {avg_precision:.4f}')
    print(f'F1-score@{k}: {avg_f1:.4f}')

    return avg_recall, avg_precision, avg_f1

def split_data(interactions, test_size=0.2):
    test_indices = np.random.choice(interactions.index, size=int(len(interactions) * test_size), replace=False)
    test = interactions.loc[test_indices]
    train = interactions.drop(test_indices)
    return train, test

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    movies_df, ratings_df = load_movielens_data()
    reviews_df_steam, items_df_steam = load_steam_data()

    print("Building graphs...")
    movielens_graph = transform_movielenses_to_graph(ratings_df.copy(), movies_df.copy())
    steam_graph = transform_steam_to_graph(reviews_df_steam, items_df_steam)
    
    print("Merging graphs...")
    merged_graph = merge_graphs(movielens_graph, steam_graph)

    # Create interactions DF for splitting
    all_interactions_df = pd.DataFrame(merged_graph['user_item_edges'], columns=['user_id', 'item_id'])
    
    unique_users = all_interactions_df['user_id'].unique()
    unique_items = all_interactions_df['item_id'].unique()
    user_map = {user: i for i, user in enumerate(unique_users)}
    item_map = {item: i for i, item in enumerate(unique_items)}
    all_interactions_df['user_idx'] = all_interactions_df['user_id'].map(user_map)
    all_interactions_df['item_idx'] = all_interactions_df['item_id'].map(item_map)

    print("Splitting data...")
    train_interactions_df, test_interactions_df = split_data(all_interactions_df)

    # Reconstruct training graph
    train_graph = {
        'user_nodes': merged_graph['user_nodes'],
        'item_nodes': merged_graph['item_nodes'],
        'genre_nodes': merged_graph['genre_nodes'],
        'user_item_edges': [tuple(x) for x in train_interactions_df[['user_id', 'item_id']].to_numpy()],
        'item_genre_edges': merged_graph['item_genre_edges'],
    }

    print("Creating PyG graph...")
    G_pyg_train, node_to_int_id, int_id_to_node = create_pyg_graph(train_graph)
    
    # --- Training ---
    print("Initializing Node2Vec...")
    model = Node2Vec(
        edge_index=G_pyg_train.edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=20,
        num_negative_samples=1,
        p=1,
        q=0.5,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0) # reduced workers for safety
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print("Starting training...")
    epochs = 10 # Set to 10 for standard run
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')
        lr_scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/node2vec_model.pth')

    print("Training complete.")
    
    # --- Evaluation ---
    model.eval()
    embeddings = model.embedding.weight.detach()
    
    evaluate_node2vec(
        embeddings, 
        test_interactions_df, 
        train_interactions_df, 
        node_to_int_id, 
        user_map, 
        item_map, 
        device, 
        k=20
    )

if __name__ == "__main__":
    main()
