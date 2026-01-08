# Generic Recommender System

## Authors

* Mateusz Bernart (156072)
* Patryk Janiak (156053)
* Marek Seget (156042)

## About The Project

This project implements a Generic Recommender System that unifies diverse datasets (e.g., MovieLens, Steam) into a single heterogeneous graph structure using a "Convert-First" strategy.

The system supports multiple backends, ranging from traditional collaborative filtering to advanced graph embeddings:
* High Accuracy Models: LightGCN, Node2Vec.
* High Novelty Models: Cleora, NCF.
* Baselines: Matrix Factorization.

## Project Structure

* `src/interfaces.py`: Defines the abstract `EmbeddingGenerator` base class.
* `src/models/`: Implementation of models (LightGCN, Cleora, Node2Vec, etc.).
* `notebooks/`: Jupyter notebooks for data exploration and full experiments.
* `helpers/`: Utilities for data loading and preprocessing.

## Setup & Installation

To run code inside of the notebooks we recommend you to create a new Python's virtual environment. We tested the code with Python version `3.10.18`. Here is an example how to setup such a virtual environment with Python already installed and avaiable system-wide, to do so we will be using `uv`.

```console
uv venv env --python 3.10
source env/bin/activate
uv pip install -r notebooks/requirements.txt
```

After running this code you will be able to download the datasets using the `dvc` command.

```console
dvc pull
```

## Usage
Below is a complete example of how to load data, merge datasets, and train the models programmatically.

```python
import sys
import os
import pandas as pd
from helpers.data_loaders import load_movielens_data, load_steam_data
from src.evaluation import leave_one_out_split, calculate_metrics, print_metrics
from src.models import (
    LightGCNGenerator, Node2VecGenerator, CleoraGenerator,
    MatrixFactorizationGenerator, NCFGenerator
)

# 1. Load & Preprocess MovieLens (Explicit -> Implicit)
movies_df, ratings_df = load_movielens_data("datasets/movies/movies.csv", "datasets/movies/ratings.csv")
movielens = pd.DataFrame({
    'user_id': 'ml_user_' + ratings_df['userId'].astype(str),
    'item_id': 'ml_item_' + ratings_df['movieId'].astype(str),
    'rating': ratings_df['rating'],
    'timestamp': ratings_df['timestamp']
})
# Filter for positive interactions (>= 4.0)
movielens = movielens[movielens['rating'] >= 4.0].copy()

# 2. Load & Preprocess Steam (Implicit)
reviews_df, _ = load_steam_data("datasets/steam/formatted_user_reviews.json", "datasets/steam/formatted_steam_games.json")
steam = pd.DataFrame({
    'user_id': 'steam_user_' + reviews_df['user_id'].astype(str),
    'item_id': 'steam_item_' + reviews_df['app_id'].astype(str),
    'rating': 1.0, 
    'timestamp': 0 
})

# 3. Merge Datasets
all_interactions = pd.concat([movielens, steam]).drop_duplicates(subset=['user_id', 'item_id'])
print(f"Total interactions: {len(all_interactions)}")

# 4. Split Data (Leave-One-Out)
train_df, test_df = leave_one_out_split(all_interactions, time_col='timestamp')

# 5. Define Models
models = [
    ("LightGCN", LightGCNGenerator(epochs=5, batch_size=128)),
    ("Node2Vec", Node2VecGenerator(epochs=100, batch_size=128)),
    ("MatrixFactorization", MatrixFactorizationGenerator(epochs=5, batch_size=128)),
    ("NCF", NCFGenerator(epochs=15, batch_size=128)),
    ("Cleora", CleoraGenerator(num_walks=3))
]

# 6. Train & Evaluate Loop
for name, model in models:    
    print(f"\n--- Running {name} ---")
    
    # Train
    print(f"Training...")
    model.fit(train_df, user_col='user_id', item_col='item_id', rating_col='rating')
    
    # Generate Embeddings
    embeddings = model.get_embeddings()
    print(f"Generated {len(embeddings)} embeddings.")

    # Evaluate
    print(f"Evaluating...")
    metrics = calculate_metrics(model, train_df, test_df)
    print_metrics(metrics)
    
    # Save
    model.save(f"models/{name.lower()}_model.pth")
```
