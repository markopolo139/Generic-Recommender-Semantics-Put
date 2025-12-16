import pandas as pd

def filter_users_by_interaction_count(df, user_col, min_count, max_count):
    """
    Filters out users that have an interaction count outside the provided range.
    """
    user_counts = df[user_col].value_counts()
    filtered_users = user_counts[(user_counts >= min_count) & (user_counts <= max_count)].index
    filtered_df = df[df[user_col].isin(filtered_users)]
    return filtered_df
