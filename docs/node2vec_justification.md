# Justification for node2vec

`node2vec` was chosen for this recommender system due to its ability to generate effective node embeddings from our heterogeneous graph (users, items, genres).

Key reasons:

1.  Our system models user-item interactions and item-genre relationships as a graph, capturing rich connections.
2.  To enable machine learning on this graph, nodes are converted into embeddings that reflect their structural position.
3.  Pros:
    - Uses biased random walks to learn both local and global neighborhood information.
    - Tunable parameters (`p`, `q`) allow for capturing similar nodes.
    - Learns embeddings from graph structure without explicit labels.
    - Effectively applies to our multi-typed graph (users, items, genres).
    - Embeddings can be used for various downstream tasks like recommendation, similarity search, and clustering.

`node2vec` provides a transforms complex graph data into actionable insights
