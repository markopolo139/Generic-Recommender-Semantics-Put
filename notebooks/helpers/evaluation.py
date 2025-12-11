import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, cdist

def leave_one_out_split(interactions_df, user_col='user_id', item_col='item_id', time_col=None):
    """
    Splits the interactions into train and test sets using Leave-One-Out strategy.
    For each user, the last interaction (by time) or a random one is used for testing.
    
    Args:
        interactions_df: DataFrame containing interactions.
        user_col: Column name for users.
        item_col: Column name for items.
        time_col: Optional column name for timestamps. If provided, sorts by time.
        
    Returns:
        train_df, test_df
    """
    interactions_df = interactions_df.copy()
    
    if time_col:
        interactions_df = interactions_df.sort_values(by=[user_col, time_col])
    else:
        # If no time, shuffle to ensure randomness if we pick last
        interactions_df = interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    # Group by user and take the last item for test
    # We want to keep all but the last one for training
    
    # A generic way:
    # Mark the last item for each user
    interactions_df['rank_order'] = interactions_df.groupby(user_col).cumcount(ascending=False)
    
    test_mask = interactions_df['rank_order'] == 0
    train_mask = ~test_mask
    
    test_df = interactions_df[test_mask].drop(columns=['rank_order'])
    train_df = interactions_df[train_mask].drop(columns=['rank_order'])
    
    return train_df, test_df

def calculate_metrics(model, train_df, test_df, user_col='user_id', item_col='item_id', k_list=[20, 50, 100, 1000], num_neg_samples=100):
    """
    Calculates evaluation metrics for the given model.
    Includes: Precision, NDCG, HitRate, MAP, AUC, Diversity, Novelty, Serendipity, Popularity Bias.
    
    Args:
        model: Trained model implementing get_embeddings().
        train_df: Training interactions (for masking and history).
        test_df: Testing interactions (ground truth).
        user_col: User column name.
        item_col: Item column name.
        k_list: List of K values for metrics.
        num_neg_samples: Number of negative samples for Sampled AUC.
        
    Returns:
        Dictionary of metrics.
    """
    embeddings = model.get_embeddings()
    
    if not embeddings:
        return {}
    
    # --- Pre-computation ---
    
    all_keys = set(embeddings.keys())
    known_users = set(train_df[user_col].unique()) | set(test_df[user_col].unique())
    known_items = set(train_df[item_col].unique()) | set(test_df[item_col].unique())
    
    # Filter embeddings for items
    valid_items = [i for i in known_items if i in embeddings]
    if not valid_items:
        raise ValueError("No item embeddings found matching the data.")
        
    item_id_to_idx = {item: i for i, item in enumerate(valid_items)}
    idx_to_item_id = {i: item for item, i in item_id_to_idx.items()}
    
    item_matrix = np.stack([embeddings[item] for item in valid_items])
    
    # Pre-compute training history for masking and behavioral metrics
    train_user_items = train_df.groupby(user_col)[item_col].apply(set).to_dict()
    
    # Item Popularity & Novelty
    item_counts = train_df[item_col].value_counts()
    total_interactions = len(train_df)
    # P(i)
    item_probs = (item_counts / total_interactions).to_dict()
    min_prob = 1.0 / total_interactions
    
    def get_self_information(item_id):
        p = item_probs.get(item_id, min_prob)
        return -np.log2(p)
    
    # Pre-compute self-information for all valid items to speed up
    item_self_info_array = np.array([get_self_information(item) for item in valid_items])
    item_popularity_array = np.array([item_counts.get(item, 0) for item in valid_items])

    # Initialize metrics storage
    metrics = {}
    metric_names = [
        "Precision", "HitRate", "NDCG", "MAP", 
        "Diversity", "Novelty", "Serendipity", "AveragePopularity"
    ]
    for m in metric_names:
        for k in k_list:
            metrics[f"{m}@{k}"] = []
            
    metrics["AUC (Global)"] = []
    metrics["AUC (Sampled)"] = []
    metrics["MeanRank"] = []
    metrics["MRR"] = []
    
    users_to_evaluate = test_df[user_col].unique()
    
    for user in tqdm(users_to_evaluate, desc="Evaluating"):
        if user not in embeddings:
            continue
            
        user_emb = embeddings[user]
        gt_items = set(test_df[test_df[user_col] == user][item_col].values)
        
        # Calculate scores for ALL items
        scores = np.dot(item_matrix, user_emb)
        
        # Mask training items
        history_items = []
        if user in train_user_items:
            history_items = list(train_user_items[user])
            train_indices = [item_id_to_idx[i] for i in history_items if i in item_id_to_idx]
            scores[train_indices] = -np.inf
            
        # Get sorted indices (descending score)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Identify GT indices
        gt_indices = [item_id_to_idx[i] for i in gt_items if i in item_id_to_idx]
        
        if not gt_indices:
            continue
            
        # --- LOO Metrics (Rank, AUC, MRR) ---
        # Optimized for LOO (single GT)
        if len(gt_indices) == 1:
            gt_idx = gt_indices[0]
            gt_score = scores[gt_idx]
            rank = np.sum(scores > gt_score) + 1
            # MeanRank: The average rank of the ground-truth item among all candidate items for each user.
            metrics["MeanRank"].append(rank)
            # MRR (Mean Reciprocal Rank): The average of the reciprocal ranks of the first relevant item for a set of queries.
            metrics["MRR"].append(1.0 / rank)
            
            # AUC
            valid_indices = np.where(scores > -np.inf)[0]
            neg_indices = valid_indices[valid_indices != gt_idx]
            
            if len(neg_indices) > 0:
                neg_scores = scores[neg_indices]
                num_worse_global = np.sum(neg_scores < gt_score)
                # AUC (Global): Area Under the ROC Curve, representing the probability that the model ranks a randomly chosen positive item higher than a randomly chosen negative item from the entire item set.
                metrics["AUC (Global)"].append(num_worse_global / len(neg_indices))
                
                if len(neg_indices) > num_neg_samples:
                    sampled_neg_indices = np.random.choice(neg_indices, num_neg_samples, replace=False)
                else:
                    sampled_neg_indices = neg_indices
                
                sampled_neg_scores = scores[sampled_neg_indices]
                num_worse_sampled = np.sum(sampled_neg_scores < gt_score)
                # AUC (Sampled): Area Under the ROC Curve, similar to global AUC but calculated on a randomly sampled subset of negative items for efficiency.
                metrics["AUC (Sampled)"].append(num_worse_sampled / len(sampled_neg_indices))
        else:
            # Fallback for multi-target (simplified, mostly reusing LOO logic per item or just skip rank/AUC specific details)
            pass

        # --- Top-K Metrics ---
        
        # User History Embedding (Mean Profile) for Serendipity
        user_history_emb = None
        if history_items:
            valid_history_items = [i for i in history_items if i in embeddings]
            if valid_history_items:
                user_history_emb = np.mean([embeddings[i] for i in valid_history_items], axis=0).reshape(1, -1)
        
        for k in k_list:
            top_k_indices = sorted_indices[:k]
            
            # Hit based metrics
            # Count intersection between top_k_indices and gt_indices
            hits = 0
            for idx in top_k_indices:
                if idx in gt_indices:
                    hits += 1
            
            is_hit = 1 if hits > 0 else 0
            # HitRate@K: Binary metric, 1 if any ground-truth item is in the top-K recommendations, 0 otherwise.
            metrics[f"HitRate@{k}"].append(is_hit)
            # Precision@K: Proportion of recommended items in the top-K that are relevant.
            metrics[f"Precision@{k}"].append(hits / k)
            
            # NDCG
            # For LOO, if hit is at rank r <= k, NDCG = 1/log2(r+1). Else 0.
            # If multiple hits, standard NDCG formula.
            dcg = 0.0
            idcg = 0.0
            
            # Calculate DCG
            for i, idx in enumerate(top_k_indices):
                if idx in gt_indices:
                    dcg += 1.0 / np.log2(i + 2)
            
            # Calculate IDCG
            num_gt = len(gt_indices)
            for i in range(min(num_gt, k)):
                idcg += 1.0 / np.log2(i + 2)
                
            if idcg > 0:
                # NDCG@K: Measures the quality of ranking by considering the position of relevant items. Higher values indicate better rankings.
                metrics[f"NDCG@{k}"].append(dcg / idcg)
            else:
                metrics[f"NDCG@{k}"].append(0.0)
                
            # MAP (Mean Average Precision)
            # AP@K = (Sum of P@i * rel(i)) / min(k, total_relevant)
            # For LOO: if rank <= k, AP = 1/rank. Else 0.
            ap = 0.0
            num_hits = 0
            for i, idx in enumerate(top_k_indices):
                if idx in gt_indices:
                    num_hits += 1
                    ap += num_hits / (i + 1)
            
            # Standard definition divides by Total Relevant Items.
            # In RecSys often min(k, total_relevant) or just total_relevant.
            # We'll use total_relevant (len(gt_indices)).
            if len(gt_indices) > 0:
                # MAP@K: Mean Average Precision at K. A measure of ranking quality, averaging the precision at each relevant item found.
                metrics[f"MAP@{k}"].append(ap / len(gt_indices))
            else:
                metrics[f"MAP@{k}"].append(0.0)

            # --- Behavioral Metrics ---
            
            # Retrieve embeddings and stats for Top K items
            top_k_embs = item_matrix[top_k_indices]
            top_k_novelty = item_self_info_array[top_k_indices]
            top_k_popularity = item_popularity_array[top_k_indices]
            
            # Novelty@K: Measures how novel the recommended items are to the user, often quantified by the inverse popularity of items.
            metrics[f"Novelty@{k}"].append(np.mean(top_k_novelty))
            
            # AveragePopularity@K: Measures the average popularity of recommended items. Higher values indicate a bias towards popular items.
            metrics[f"AveragePopularity@{k}"].append(np.mean(top_k_popularity))
            
            # Diversity (Intra-List)
            # pairwise cosine distance
            if k > 1:
                # Diversity@K (Intra-List Diversity): Measures how dissimilar the recommended items are to each other within the top-K list.
                div = np.mean(pdist(top_k_embs, metric='cosine'))
                metrics[f"Diversity@{k}"].append(div)
            else:
                metrics[f"Diversity@{k}"].append(0.0) # Undefined/Zero for list size 1
                
            # Serendipity
            # Mean distance of Top K items to User History Profile
            if user_history_emb is not None:
                # Serendipity@K: Measures how unexpected and relevant the recommended items are to the user, beyond what's inferred from their past interactions.
                dists = cdist(top_k_embs, user_history_emb, metric='cosine')
                metrics[f"Serendipity@{k}"].append(np.mean(dists))
            else:
                # If no history, Serendipity is undefined or 1 (max surprise)? 
                # Let's use NaN or omit. Appending 0 might imply similarity.
                # Appending 1 might imply max difference.
                # Let's append NaN and ignore in mean, or just 1 (totally new/different).
                # To be safe, let's use 1.0 (assuming history is empty, everything is novel/unexpected).
                metrics[f"Serendipity@{k}"].append(1.0)

    # Average all metrics
    avg_metrics = {}
    for k, v in metrics.items():
        if v:
            # Handle NaNs if any
            clean_v = [x for x in v if not np.isnan(x)]
            if clean_v:
                avg_metrics[k] = np.mean(clean_v)
            else:
                avg_metrics[k] = 0.0
        else:
            avg_metrics[k] = 0.0
            
    return avg_metrics

def print_metrics(metrics):
    print("\nEvaluation Results:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)