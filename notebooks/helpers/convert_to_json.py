import json
import os
import ast
import argparse

def format_steam_games(input_path, output_path):
    print(f"Formatting {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('[')
        first_game = True
        for line in infile:
            try:
                game_data = ast.literal_eval(line)
                if not first_game:
                    outfile.write(',')
                formatted_game = {
                    'app_id': game_data.get('id'),
                    'title': game_data.get('title'),
                    'genres': game_data.get('genres')
                }
                json.dump(formatted_game, outfile)
                if first_game:
                    first_game = False
            except json.JSONDecodeError:
                print(f"Skipping line due to JSONDecodeError: {line}")
        outfile.write(']')
    print(f"Formatted steam games saved to {output_path}")

def format_user_reviews(input_path, output_path):
    print(f"Formatting {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('[')
        first_review = True
        for line in infile:
            try:
                review_data = ast.literal_eval(line)
                user_reviews = review_data.get('reviews', [])
                for review in user_reviews:
                    if review.get('recommend'):
                        if not first_review:
                            outfile.write(',')
                        formatted_review = {
                            'user_id': review_data.get('user_id'),
                            'app_id': review.get('item_id')
                        }
                        json.dump(formatted_review, outfile)
                        if first_review:
                            first_review = False
            except (ValueError, SyntaxError) as e:
                print(f"Skipping line due to error: {e}")
        outfile.write(']')
    print(f"Formatted user reviews saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to format Steam game and user review data."
    )
    
    parser.add_argument(
        "working_directory", 
        type=str, 
        help="The path to the directory containing the input JSON files."
    )
    
    args = parser.parse_args()
    working_directory = args.working_directory

    steam_games_input = os.path.join(working_directory, 'steam_games.json')
    steam_games_output = os.path.join(working_directory, 'formatted_steam_games.json')
    format_steam_games(steam_games_input, steam_games_output)

    user_reviews_input = os.path.join(working_directory, 'australian_user_reviews.json')
    user_reviews_output = os.path.join(working_directory, 'formatted_user_reviews.json')
    format_user_reviews(user_reviews_input, user_reviews_output)
