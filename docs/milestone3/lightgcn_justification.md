# Justification for LightGCN

`LightGCN` was chosen for our recommender system as it represents a state-of-the-art approach in Graph Neural Network (GNN) based recommendation, known for its simplicity and strong performance.

Key reasons:

1.  Our system's user-item interactions are naturally represented as a bipartite graph, making GNNs an ideal modeling choice.
2.  `LightGCN` simplifies traditional GCNs by removing feature transformations and non-linear activation functions, which have been shown to be less effective for collaborative filtering tasks.
3.  The model learns user and item embeddings by linearly propagating them on the user-item interaction graph, effectively capturing the collaborative signal.

Pros:
    - Captures high-order connectivity in the graph, allowing it to find users with similar second-degree (and higher) neighbors.
    - Its simple design makes it computationally efficient and less prone to overfitting compared to more complex GCNs.
    - Achieves state-of-the-art results on many recommendation benchmarks.
    - The "featureless" nature (relying only on interactions) makes it a perfect fit for our heterogeneous dataset where feature engineering is complex.

`LightGCN` provides a powerful and elegant way to learn from the rich structure of user-item interactions.
