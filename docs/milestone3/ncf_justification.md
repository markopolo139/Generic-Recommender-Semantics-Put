# Justification for Neural Collaborative Filtering (NCF)

Neural Collaborative Filtering (NCF) was chosen because it extends the ideas of Matrix Factorization with deep neural networks, allowing it to capture more complex and non-linear patterns in the data.

Key reasons:

1.  While Matrix Factorization is effective, its simple dot product interaction function can be a limitation. NCF replaces this with a more expressive neural network architecture.
2.  NCF is designed with a multi-path architecture that combines two sub-models:
    - **Generalized Matrix Factorization (GMF):** A neural network-based implementation of matrix factorization that uses element-wise multiplication.
    - **Multi-Layer Perceptron (MLP):** A standard feed-forward neural network that can learn arbitrary functions and thus capture non-linear relationships between user and item embeddings.
3.  By fusing the outputs of the GMF and MLP paths, NCF can leverage both the linearity of matrix factorization and the non-linearity of a deep model.

Pros:
    - More expressive than traditional matrix factorization, with the potential for higher accuracy.
    - Capable of modeling complex, non-linear user-item interactions that linear models might miss.
    - The framework is flexible and can be extended with different neural network architectures.
    - Well-suited for implicit feedback datasets, which constitute a large part of our combined data.

NCF represents a modern approach to collaborative filtering that combines the strengths of classic methods with the power of deep learning.
