# Recommender Models Evaluation

This document summarizes the evaluation methodology and results for the four recommender models implemented: Matrix Factorization, Neural Collaborative Filtering (NCF), LightGCN, and Node2Vec.

## General Methodology

Across all models, a consistent evaluation strategy was employed to ensure fair comparison.

1.  Combined Dataset: The MovieLens and Steam datasets were combined to create a unified set of user-item interactions. For models requiring positive interactions (LightGCN, NCF), only MovieLens ratings of 4.0 or higher were included, while all Steam interactions were treated as positive implicit feedback. For Matrix Factorization, explicit ratings were used from MovieLens, and Steam interactions were given a rating of 1.0.
2.  Train-Test Split: The combined interaction data was split into a training set (80%) and a testing set (20%). The models were trained exclusively on the training set.
3.  Evaluation Metrics: The models were evaluated on their ability to predict the unseen interactions in the test set. The primary metrics used were:
    - Precision@20: The proportion of the top-20 recommended items that are relevant (i.e., are in the test set).
    - Recall@20: The proportion of relevant items in the test set that were successfully recommended in the top-20 list.
    - F1-score@20: The harmonic mean of Precision and Recall.
    - AUC (Area Under the ROC Curve): Measures the model's ability to distinguish between positive and negative items. For a given user, it's the probability that the model ranks a random positive item from the test set higher than a random negative item (an item the user has never interacted with).

## Model-Specific Evaluation

### 1. Matrix Factorization

- Approach: This model learns user and item embeddings by factorizing the user-item interaction matrix. The prediction for a user-item pair is the dot product of their respective embeddings. The model was trained using Mean Squared Error (MSE) loss on the explicit ratings.
- Evaluation: The model was evaluated using Precision@20, Recall@20, F1-score@20, and AUC. For AUC, each user's positive interactions in the test set were compared against 100 randomly sampled negative interactions. For the other metrics, the top-20 recommendations for each user were compared against the held-out items in the test set.

### 2. Neural Collaborative Filtering (NCF)

- Approach: NCF combines a Generalized Matrix Factorization (GMF) path and a Multi-Layer Perceptron (MLP) path to model user-item interactions. This allows it to capture both linear and non-linear relationships. The model was trained with Binary Cross-Entropy (BCE) loss, treating the problem as a binary classification task (interaction vs. no interaction).
- Evaluation: The notebook defines functions for calculating Precision, Recall, F1-score, and AUC. The evaluation process is standard: for each test user, predict scores for all items, exclude items seen during training, and compare the top-20 recommendations against the ground truth in the test set.

### 3. LightGCN

- Approach: LightGCN is a graph-based model that learns user and item embeddings by performing message passing on the user-item interaction graph. This allows it to capture higher-order collaborative signals. The model was trained using Bayesian Personalized Ranking (BPR) loss, which optimizes for ranking relevant items higher than irrelevant ones.
- Evaluation: The evaluation was performed using Precision@20, Recall@20, F1-score@20, and AUC, following the general methodology described above.

### 4. Node2Vec

- Approach: This model learns embeddings for nodes in a heterogeneous graph. We constructed a graph containing users, items, and genres, with edges representing user-item interactions and item-genre relationships. Node2Vec generates node embeddings by simulating random walks on this graph. The score for a user-item pair is the dot product of their learned embeddings.
- Evaluation: The model was trained on a graph containing only the training edges. The evaluation was then performed using the standard methodology for Precision@20, Recall@20, F1-score@20, and AUC on the held-out test edges.

## Results Summary

| Model                | Precision@20 | Recall@20  | F1-Score@20 | AUC        |
| -------------------- | ------------ | ---------- | ----------- | ---------- |
| Node2Vec             | 0.1161       | 0.2436     | 0.1214      | **0.9909** |
| NCF                  | **0.1341**   | **0.2704** | **0.1417**  | 0.9892     |
| LightGCN             | 0.0783       | 0.1564     | 0.0813      | 0.9783     |
| Matrix Factorization | 0.0517       | 0.0789     | 0.0429      | 0.9431     |
