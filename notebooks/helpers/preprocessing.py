import pandas as pd

def filter_users_by_interaction_count(df, user_col, min_count, max_count):
    """
    Filters out users that have an interaction count outside the provided range.
    """
    user_counts = df[user_col].value_counts()
    filtered_users = user_counts[(user_counts >= min_count) & (user_counts <= max_count)].index
    filtered_df = df[df[user_col].isin(filtered_users)]
    return filtered_df

def harmonize_genres(df, dataset_name):
    """
    Standardizes genre names to increase overlap between datasets.
    """
    if dataset_name == 'movies':
        # Mappings for Movies
        genre_map = {
            'Children': 'Casual',
            'War': 'Strategy'
        }
        # Update genres list in each row
        # genres column is a list of strings
        def update_list(genre_list):
             if isinstance(genre_list, list):
                return [genre_map.get(g, g) for g in genre_list]
             return genre_list
        
        df['genres'] = df['genres'].apply(update_list)
        
    elif dataset_name == 'steam':
        # Mappings for Steam
        genre_map = {
            'Animation & Modeling': 'Animation',
            'RPG': 'Adventure',
            'Racing': 'Action'
        }
        def update_list(genre_list):
            if isinstance(genre_list, list):
                return [genre_map.get(g, g) for g in genre_list]
            return genre_list

        df['genres'] = df['genres'].apply(update_list)
        
    return df
