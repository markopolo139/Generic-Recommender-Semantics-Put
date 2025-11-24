# Justification for Matrix Factorization

Matrix Factorization (MF) was selected as it is a foundational and highly effective collaborative filtering technique. It serves as a powerful and efficient baseline for our recommender system.

Key reasons:

1.  MF is a classic model for recommendation that learns latent factors (embeddings) for both users and items from their past interactions.
2.  The core idea is to decompose the large, sparse user-item interaction matrix into two smaller, dense matrices representing user and item embeddings.
3.  The predicted rating for an item by a user is simply the dot product of their respective embeddings, making it a very efficient model for prediction.

Pros:
    - Simple to implement, understand, and debug.
    - Computationally efficient and scales well to large datasets.
    - Provides a strong baseline to evaluate the performance of more complex models.
    - Effectively captures the underlying latent structure in the user-item interaction data.
    - Can be adapted for both explicit feedback (e.g., MovieLens ratings) and implicit feedback (e.g., Steam interactions).

Matrix Factorization is an essential starting point for building a recommender system, providing valuable insights and a solid performance benchmark.
