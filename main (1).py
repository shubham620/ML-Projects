import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Note: This script assumes you have 'movies.csv' and 'ratings.csv' in the same directory.
try:
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
except FileNotFoundError:
    print("Error: 'movies.csv' or 'ratings.csv' not found.")
    print("Please make sure the dataset files are in the same directory as the script.")
    exit()

# The title column is stripped of whitespace and converted to lowercase.
movies_df['title'] = movies_df['title'].str.strip().str.lower()
# The genres column is filled with an empty string for any missing values and then cleaned.
movies_df['genres'] = movies_df['genres'].fillna('').astype(str).str.strip().str.lower()

# Use CountVectorizer to convert the genres string into a matrix of token counts.
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(movies_df['genres'])

# Calculate the cosine similarity between all movies based on their genres.
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

def recommend_movies(title, cosine_sim_matrix=cosine_sim):
    """
    Recommends top 10 similar movies based on genre similarity.
    """
    title = title.strip().lower()

    # Create a mapping from movie titles to their index
    indices = pd.Series(movies_df.index, index=movies_df['title'])

    # Check if the movie title exists in our dataset
    if title not in indices:
        print(f"Movie title '{title}' not found.")
        print("Try using the full title, e.g., 'toy story (1995)'")
        return []

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the movies based on the similarity scores
    # The key for sorting is the similarity score (the second element of the tuple)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    return movies_df['title'].iloc[movie_indices].values

print("--- Movie Recommendation Example ---")
# Get recommendations for 'jumanji (1995)'
recommendations = recommend_movies('jumanji (1995)')
if len(recommendations) > 0:
    print("\nRecommendations for 'jumanji (1995)':")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")
print("-" * 35)

# Optional: To help users find titles, you can print a sample of unique movie titles
print("\n--- Sample of Available Movie Titles ---")
print(movies_df['title'].unique()[:20])
print("-" * 35)