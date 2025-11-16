import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def load_movielens_data(
    movies_path="datasets/movies/movies.csv",
    ratings_path="datasets/movies/ratings.csv",
):
    """
    Loads and preprocesses the MovieLens dataset.
    """
    movies_df = pd.read_csv(movies_path)
    movies_df = movies_df.dropna(subset=["title", "genres"], how="all")

    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df.dropna(subset=["movieId", "rating"], how="all")

    return movies_df, ratings_df


def load_steam_data(
    reviews_path="datasets/steam/formatted_user_reviews.json",
    items_path="datasets/steam/formatted_steam_games.json",
):
    """
    Loads and preprocesses the Steam dataset.
    """
    try:
        reviews_df_steam = pd.read_json(reviews_path)
        reviews_df_steam["app_id"] = reviews_df_steam["app_id"].astype(int)

        items_df_steam = pd.read_json(items_path)
        items_df_steam = items_df_steam.dropna(subset=["app_id"])
        items_df_steam["app_id"] = items_df_steam["app_id"].astype(int)

        items_df_steam_cleaned = items_df_steam.dropna(
            subset=["title", "genres"], how="all"
        )

        reviews_df_steam_cleaned = reviews_df_steam[
            reviews_df_steam["app_id"].isin(items_df_steam_cleaned["app_id"].tolist())
        ]

        return reviews_df_steam_cleaned, items_df_steam_cleaned
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure the file paths are correct.")
        return None, None


def transform_movielenses_to_graph(
    ratings_df, movies_df, rating_threshold=4.0, dataset_name="MovieLens"
):
    recommended_df = ratings_df[ratings_df["rating"] >= rating_threshold].copy()

    recommended_df["userId"] = (
        f"{dataset_name}_user_" + recommended_df["userId"].astype(str)
    )
    movies_df["movieId"] = f"{dataset_name}_item_" + movies_df["movieId"].astype(str)

    movies_df["genres"] = movies_df["genres"].str.split("|")

    user_nodes = recommended_df["userId"].unique()
    item_nodes = movies_df["movieId"].unique()

    user_item_edges = list(
        zip(
            recommended_df["userId"],
            f"{dataset_name}_item_" + recommended_df["movieId"].astype(str),
        )
    )

    item_genre_edges = []
    genre_nodes = set()
    exploded_genres = movies_df.explode("genres")
    for _, row in exploded_genres.iterrows():
        genre = str(row["genres"]).lower().replace(" ", "_")
        item_genre_edges.append((row["movieId"], genre))
        genre_nodes.add(genre)

    return {
        "user_nodes": list(user_nodes),
        "item_nodes": list(item_nodes),
        "genre_nodes": list(genre_nodes),
        "user_item_edges": user_item_edges,
        "item_genre_edges": item_genre_edges,
    }


def transform_steam_to_graph(reviews_df, items_df, dataset_name="Steam"):
    user_item_edges = list(
        zip(
            f"{dataset_name}_user_" + reviews_df["user_id"].astype(str),
            f"{dataset_name}_item_" + reviews_df["app_id"].astype(str),
        )
    )

    item_genre_edges = []
    genre_nodes = set()

    items_df_exploded = items_df.explode("genres")
    for _, row in items_df_exploded.iterrows():
        genre = str(row["genres"]).lower().replace(" ", "_")
        if genre:
            item_genre_edges.append((f"{dataset_name}_item_{row['app_id']}", genre))
            genre_nodes.add(genre)

    user_nodes = list(set([edge[0] for edge in user_item_edges]))
    item_nodes = list(set([edge[1] for edge in user_item_edges]))

    return {
        "user_nodes": user_nodes,
        "item_nodes": item_nodes,
        "genre_nodes": list(genre_nodes),
        "user_item_edges": user_item_edges,
        "item_genre_edges": item_genre_edges,
    }


def merge_graphs(graph1, graph2):
    merged_graph = {
        "user_nodes": list(set(graph1["user_nodes"] + graph2["user_nodes"])),
        "item_nodes": list(set(graph1["item_nodes"] + graph2["item_nodes"])),
        "genre_nodes": list(set(graph1["genre_nodes"] + graph2["genre_nodes"])),
        "user_item_edges": list(
            set(graph1["user_item_edges"] + graph2["user_item_edges"])
        ),
        "item_genre_edges": list(
            set(graph1["item_genre_edges"] + graph2["item_genre_edges"])
        ),
    }
    return merged_graph


def create_pyg_graph(merged_graph):
    all_nodes_list = sorted(
        list(
            set(merged_graph["user_nodes"])
            | set(merged_graph["item_nodes"])
            | set(merged_graph["genre_nodes"])
        )
    )

    node_to_int_id = {node: i for i, node in enumerate(all_nodes_list)}
    num_nodes = len(all_nodes_list)

    source_edges = []
    target_edges = []

    all_edges = merged_graph["user_item_edges"] + merged_graph["item_genre_edges"]

    for u, v in all_edges:
        if u in node_to_int_id and v in node_to_int_id:
            u_id = node_to_int_id[u]
            v_id = node_to_int_id[v]

            source_edges.append(u_id)
            target_edges.append(v_id)
            source_edges.append(v_id)
            target_edges.append(u_id)


    edge_index = torch.tensor([source_edges, target_edges], dtype=torch.long)
    G_pyg = Data(num_nodes=num_nodes, edge_index=edge_index)

    int_id_to_node = {i: node for node, i in node_to_int_id.items()}

    return G_pyg, node_to_int_id, int_id_to_node