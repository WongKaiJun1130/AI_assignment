# ============================================================
# MOVIE RECOMMENDER SYSTEM
# Methods:
# 1. Popularity-Based Recommendation
# 2. Content-Based Filtering
# 3. Collaborative Filtering
# 4. Hybrid Recommendation
#
# Required dataset files:
# movies_metadata.csv
# ratings_small.csv
# links_small.csv
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import ast
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Import Surprise Library for Collaborative Filtering
try:
    from surprise import Dataset, Reader, SVD
except ImportError as error:
    raise ImportError("Please install scikit-surprise first: pip install scikit-surprise") from error

# Hide unnecessary warning messages
warnings.filterwarnings("ignore")

# Display more columns when printing DataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


# ============================================================
# 2. FILE SETTINGS AND VALIDATION
# ============================================================

# Dataset file names
MOVIES_FILE = "movies_metadata.csv"
RATINGS_FILE = "ratings_small.csv"
LINKS_FILE = "links_small.csv"

# List of files required by the program
REQUIRED_FILES = [MOVIES_FILE, RATINGS_FILE, LINKS_FILE]

# Check whether all required files exist
missing_files = [file_name for file_name in REQUIRED_FILES if not os.path.exists(file_name)]

# Stop the program if a dataset file is missing
if missing_files:
    raise FileNotFoundError(f"Missing dataset file(s): {missing_files}. Put all CSV files in the same folder as this notebook or Python file.")


# ============================================================
# 3. LOAD DATASETS
# ============================================================

# Load movie metadata
metadata = pd.read_csv(MOVIES_FILE, low_memory=False)

# Load user rating records
ratings = pd.read_csv(RATINGS_FILE)

# Load MovieLens and TMDB ID mapping
links = pd.read_csv(LINKS_FILE)

# Display the original dataset sizes
print("Original metadata shape:", metadata.shape)
print("Original ratings shape:", ratings.shape)
print("Original links shape:", links.shape)


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

def extract_genres(value):
    """
    Convert the JSON-like genres column into simple text.

    Example:
    [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]

    Output:
    Animation Comedy
    """

    try:
        genre_list = ast.literal_eval(value) if isinstance(value, str) else []
        return " ".join([genre.get("name", "").replace(" ", "") for genre in genre_list])

    except (ValueError, SyntaxError, TypeError):
        return ""


# Convert movie metadata ID into numeric TMDB ID
metadata["tmdbId"] = pd.to_numeric(metadata["id"], errors="coerce")

# Convert vote columns into numeric values
metadata["vote_count"] = pd.to_numeric(metadata["vote_count"], errors="coerce")
metadata["vote_average"] = pd.to_numeric(metadata["vote_average"], errors="coerce")

# Convert link IDs into numeric values
links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")

# Convert rating columns into numeric values
ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce")
ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

# Remove metadata records without a valid ID or title
metadata = metadata.dropna(subset=["tmdbId", "title"]).copy()

# Remove link records without valid IDs
links = links.dropna(subset=["movieId", "tmdbId"]).copy()

# Remove rating records without user ID, movie ID, or rating
ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()

# Convert identifiers into integer data type
metadata["tmdbId"] = metadata["tmdbId"].astype(int)
links["movieId"] = links["movieId"].astype(int)
links["tmdbId"] = links["tmdbId"].astype(int)
ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

# Convert rating into float
ratings["rating"] = ratings["rating"].astype(float)

# Replace missing movie overview with empty text
metadata["overview"] = metadata["overview"].fillna("")

# Replace missing genres with empty JSON list
metadata["genres"] = metadata["genres"].fillna("[]")

# Replace missing vote information with zero
metadata["vote_count"] = metadata["vote_count"].fillna(0)
metadata["vote_average"] = metadata["vote_average"].fillna(0)

# Extract genre names
metadata["genres_text"] = metadata["genres"].apply(extract_genres)

# Combine movie overview and genres into one content feature
metadata["content_text"] = metadata["overview"].astype(str) + " " + metadata["genres_text"].astype(str)

# Keep only columns required by the recommender system
metadata = metadata[["tmdbId", "title", "overview", "genres_text", "content_text", "vote_count", "vote_average"]].copy()

# Remove duplicate TMDB movie records
metadata = metadata.drop_duplicates(subset=["tmdbId"]).reset_index(drop=True)

# Connect MovieLens movieId with TMDB metadata
movie_data = links[["movieId", "tmdbId"]].merge(metadata, on="tmdbId", how="inner")

# Remove duplicate MovieLens movie records
movie_data = movie_data.drop_duplicates(subset=["movieId"]).reset_index(drop=True)

# Keep ratings only when the movie exists in movie_data
ratings = ratings[ratings["movieId"].isin(movie_data["movieId"])].copy()

# Keep the latest rating if the same user rated the same movie more than once
ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last").reset_index(drop=True)

# Add movie title into rating records
full_data = ratings.merge(movie_data[["movieId", "title"]], on="movieId", how="inner")

# Stop the program when no valid data remains
if movie_data.empty or ratings.empty:
    raise ValueError("No valid movie or rating records remain after preprocessing. Please check the dataset files.")

# Display cleaned dataset information
print("\nMatched movies:", movie_data["movieId"].nunique())
print("Matched users:", ratings["userId"].nunique())
print("Matched ratings:", len(ratings))


# ============================================================
# 5. PRELIMINARY DATA ANALYSIS
# ============================================================

# Calculate the number of users
number_of_users = ratings["userId"].nunique()

# Calculate the number of movies
number_of_movies = ratings["movieId"].nunique()

# Calculate the number of rating records
number_of_ratings = len(ratings)

# Calculate the total possible number of ratings
possible_ratings = number_of_users * number_of_movies

# Calculate user-item matrix sparsity
matrix_sparsity = 1 - (number_of_ratings / possible_ratings)

# Display dataset statistics
print("\nAverage user rating:", round(ratings["rating"].mean(), 4))
print("Minimum user rating:", ratings["rating"].min())
print("Maximum user rating:", ratings["rating"].max())
print("User-item matrix sparsity:", f"{matrix_sparsity:.2%}")

# Display missing values
print("\nMissing values in movie data:")
print(movie_data.isnull().sum())


# ============================================================
# 6. RATING DISTRIBUTION CHART
# ============================================================

# Create rating distribution chart
plt.figure(figsize=(8, 5))
ratings["rating"].hist(bins=10, edgecolor="black")
plt.title("Distribution of User Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.show()


# ============================================================
# 7. TOP 10 MOST-RATED MOVIES CHART
# ============================================================

# Count the number of ratings received by each movie
most_rated_movies = full_data.groupby("title")["rating"].count().sort_values(ascending=False).head(10)

# Create horizontal bar chart
plt.figure(figsize=(10, 5))
most_rated_movies.sort_values().plot(kind="barh")
plt.title("Top 10 Most-Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie Title")
plt.tight_layout()
plt.show()


# ============================================================
# 8. METHOD 1: POPULARITY-BASED RECOMMENDER
# ============================================================

# Calculate the average vote score across all movies
C = movie_data["vote_average"].mean()

# Calculate the minimum vote count using the 90th percentile
m = movie_data["vote_count"].quantile(0.90)


def weighted_rating(row, minimum_votes=m, average_score=C):
    """
    Calculate IMDb-style weighted rating.

    v = Number of votes received by the movie
    R = Average rating of the movie
    m = Minimum votes required
    C = Average rating across all movies
    """

    vote_count = row["vote_count"]
    vote_average = row["vote_average"]

    # Avoid division by zero
    if vote_count + minimum_votes == 0:
        return 0.0

    # Return the weighted rating
    return (vote_count / (vote_count + minimum_votes) * vote_average) + (minimum_votes / (vote_count + minimum_votes) * average_score)


# Calculate weighted popularity score for every movie
movie_data["weighted_score"] = movie_data.apply(weighted_rating, axis=1)


def recommend_by_popularity(top_n=10):
    """
    Return the most popular movies.

    This method gives the same recommendation to all users.
    It is suitable for new users without rating history.
    """

    # Keep movies that have enough votes
    qualified_movies = movie_data[movie_data["vote_count"] >= m].copy()

    # Sort movies from highest to lowest weighted score
    qualified_movies = qualified_movies.sort_values(by="weighted_score", ascending=False)

    # Return the Top N movies
    return qualified_movies[["movieId", "title", "vote_count", "vote_average", "weighted_score"]].head(top_n).reset_index(drop=True)


# ============================================================
# 9. METHOD 2: CONTENT-BASED FILTERING
# ============================================================

# Keep movies that contain useful content text
content_movies = movie_data[movie_data["content_text"].str.strip() != ""].copy()

# Remove duplicated movie titles
content_movies = content_movies.drop_duplicates(subset=["title"]).reset_index(drop=True)

# Create TF-IDF Vectorizer
content_vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)

# Convert movie overview and genre text into TF-IDF vectors
content_matrix = content_vectorizer.fit_transform(content_movies["content_text"])

# Create a nearest-neighbour model using cosine similarity
content_model = NearestNeighbors(metric="cosine", algorithm="brute")

# Train the content-based model
content_model.fit(content_matrix)

# Create a movieId-to-index mapping
content_index_by_movie_id = pd.Series(content_movies.index, index=content_movies["movieId"]).to_dict()


def find_movie_title(title):
    """
    Find the index of a movie title.

    The function tries:
    1. Exact title match
    2. Partial title match
    """

    # Search using exact title
    exact_match = content_movies[content_movies["title"].str.lower() == title.strip().lower()]

    if not exact_match.empty:
        return exact_match.index[0]

    # Search using part of the title
    partial_match = content_movies[content_movies["title"].str.contains(title.strip(), case=False, regex=False)]

    if not partial_match.empty:
        return partial_match.index[0]

    # Display error when movie is not found
    raise ValueError(f"Movie title not found: {title}")


def recommend_by_content(title, top_n=10):
    """
    Recommend movies with similar overview and genre information.
    """

    # Find the selected movie index
    movie_index = find_movie_title(title)

    # Make sure requested neighbours do not exceed dataset size
    number_of_neighbors = min(top_n + 1, len(content_movies))

    # Find nearest movie vectors
    distances, indices = content_model.kneighbors(content_matrix[movie_index], n_neighbors=number_of_neighbors)

    # Remove the first result because it is the selected movie itself
    recommendation_indices = indices.flatten()[1:]

    # Convert cosine distance into cosine similarity
    similarity_scores = 1 - distances.flatten()[1:]

    # Obtain recommended movie information
    recommendations = content_movies.iloc[recommendation_indices][["movieId", "title", "genres_text", "vote_average", "weighted_score"]].copy()

    # Add content similarity score
    recommendations["content_score"] = similarity_scores

    # Return Top N recommendations
    return recommendations[["movieId", "title", "genres_text", "vote_average", "content_score", "weighted_score"]].head(top_n).reset_index(drop=True)


# ============================================================
# 10. SPLIT RATING DATA
# ============================================================

# Use 80% of ratings for training and 20% for testing
train_ratings, test_ratings = train_test_split(ratings, test_size=0.20, random_state=42)

# Calculate global average rating from training data
global_rating_mean = train_ratings["rating"].mean()

# Group each user's training ratings for content prediction
user_training_groups = {int(user_id): group[["movieId", "rating"]].copy() for user_id, group in train_ratings.groupby("userId")}


# ============================================================
# 11. METHOD 3: COLLABORATIVE FILTERING USING SVD
# ============================================================

# Define the valid rating range
rating_reader = Reader(rating_scale=(0.5, 5.0))

# Convert Pandas DataFrame into Surprise Dataset
surprise_dataset = Dataset.load_from_df(train_ratings[["userId", "movieId", "rating"]], rating_reader)

# Build the full training set
surprise_trainset = surprise_dataset.build_full_trainset()

# Create the SVD Collaborative Filtering model
collaborative_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

# Train the Collaborative Filtering model
collaborative_model.fit(surprise_trainset)


def recommend_by_collaborative(user_id, top_n=10):
    """
    Recommend unseen movies using predicted user ratings.
    """

    # Get movies already rated by the user
    rated_movie_ids = set(ratings[ratings["userId"] == user_id]["movieId"])

    # Keep movies that have not been rated by the user
    unseen_movies = movie_data[~movie_data["movieId"].isin(rated_movie_ids)][["movieId", "title", "genres_text", "weighted_score"]].copy()

    # Predict the user's rating for every unseen movie
    unseen_movies["predicted_rating"] = [collaborative_model.predict(user_id, int(movie_id)).est for movie_id in unseen_movies["movieId"]]

    # Sort movies from highest to lowest predicted rating
    unseen_movies = unseen_movies.sort_values(by="predicted_rating", ascending=False)

    # Return Top N recommendations
    return unseen_movies[["movieId", "title", "genres_text", "predicted_rating", "weighted_score"]].head(top_n).reset_index(drop=True)


# ============================================================
# 12. CONTENT-BASED RATING PREDICTION
# ============================================================

def predict_content_rating(user_id, movie_id):
    """
    Predict a user's rating using content similarity.

    The function compares the target movie with movies
    previously rated by the user.
    """

    # Use global average when movie has no content vector
    if movie_id not in content_index_by_movie_id:
        return global_rating_mean

    # Obtain the user's previous rating history
    user_history = user_training_groups.get(int(user_id))

    # Use global average when user has no rating history
    if user_history is None or user_history.empty:
        return global_rating_mean

    # Keep user-rated movies that have content vectors
    usable_history = user_history[user_history["movieId"].isin(content_index_by_movie_id)].copy()

    # Use the user's average when none of the rated movies has content vectors
    if usable_history.empty:
        return float(user_history["rating"].mean())

    # Get the target movie content index
    target_index = content_index_by_movie_id[int(movie_id)]

    # Get content indices of movies previously rated by the user
    history_indices = [content_index_by_movie_id[int(history_movie_id)] for history_movie_id in usable_history["movieId"]]

    # Calculate cosine similarity between target movie and user-rated movies
    similarities = linear_kernel(content_matrix[target_index], content_matrix[history_indices]).flatten()

    # Keep positive similarities
    positive_mask = similarities > 0

    # Use the user's average when no similar movie exists
    if not positive_mask.any():
        return float(user_history["rating"].mean())

    # Calculate weighted average rating using content similarities
    predicted_rating = np.average(usable_history["rating"].to_numpy()[positive_mask], weights=similarities[positive_mask])

    # Keep prediction within the valid rating scale
    return float(np.clip(predicted_rating, 0.5, 5.0))


# Create quick movieId-to-popularity-score mapping
popularity_score_by_movie_id = pd.Series(movie_data["weighted_score"].values, index=movie_data["movieId"]).to_dict()


def predict_popularity_rating(movie_id):
    """
    Convert the 0-to-10 popularity score into the 0.5-to-5 rating scale.
    """

    # Get movie popularity score
    weighted_score = popularity_score_by_movie_id.get(int(movie_id), C)

    # Convert rating from scale 10 to scale 5
    return float(np.clip(weighted_score / 2, 0.5, 5.0))


# ============================================================
# 13. METHOD 4: HYBRID RECOMMENDER
# ============================================================

def recommend_by_hybrid(user_id, selected_title, top_n=10, content_weight=0.40, collaborative_weight=0.45, popularity_weight=0.15):
    """
    Combine:
    40% Content-Based Score
    45% Collaborative Filtering Score
    15% Popularity Score
    """

    # Get 50 content-similar candidate movies
    candidates = recommend_by_content(selected_title, top_n=50)

    # Find movies already rated by the user
    rated_movie_ids = set(ratings[ratings["userId"] == user_id]["movieId"])

    # Remove movies already rated by the user
    candidates = candidates[~candidates["movieId"].isin(rated_movie_ids)].copy()

    # Calculate Collaborative Filtering score from 0 to 1
    candidates["collaborative_score"] = [collaborative_model.predict(user_id, int(movie_id)).est / 5.0 for movie_id in candidates["movieId"]]

    # Keep content score between 0 and 1
    candidates["content_normalized"] = candidates["content_score"].clip(0, 1)

    # Convert popularity score into 0-to-1 scale
    candidates["popularity_normalized"] = (candidates["weighted_score"] / 10.0).clip(0, 1)

    # Calculate final Hybrid score
    candidates["hybrid_score"] = (content_weight * candidates["content_normalized"]) + (collaborative_weight * candidates["collaborative_score"]) + (popularity_weight * candidates["popularity_normalized"])

    # Sort movies from highest to lowest Hybrid score
    candidates = candidates.sort_values(by="hybrid_score", ascending=False)

    # Return Top N recommendations
    return candidates[["movieId", "title", "genres_text", "content_score", "collaborative_score", "popularity_normalized", "hybrid_score"]].head(top_n).reset_index(drop=True)


# ============================================================
# 14. OFFLINE MODEL EVALUATION
# ============================================================

# Use at most 5,000 test records to reduce evaluation time
EVALUATION_SAMPLE_SIZE = min(5000, len(test_ratings))

# Randomly select evaluation records
evaluation_data = test_ratings.sample(n=EVALUATION_SAMPLE_SIZE, random_state=42).copy().reset_index(drop=True)

print("\nGenerating evaluation predictions. This may take a few minutes...")

# Generate Popularity-Based predictions
evaluation_data["popularity_prediction"] = [predict_popularity_rating(movie_id) for movie_id in evaluation_data["movieId"]]

# Generate Content-Based predictions
evaluation_data["content_prediction"] = [predict_content_rating(user_id, movie_id) for user_id, movie_id in zip(evaluation_data["userId"], evaluation_data["movieId"])]

# Generate Collaborative Filtering predictions
evaluation_data["collaborative_prediction"] = [collaborative_model.predict(int(user_id), int(movie_id)).est for user_id, movie_id in zip(evaluation_data["userId"], evaluation_data["movieId"])]

# Generate Hybrid predictions
evaluation_data["hybrid_prediction"] = (0.15 * evaluation_data["popularity_prediction"]) + (0.40 * evaluation_data["content_prediction"]) + (0.45 * evaluation_data["collaborative_prediction"])

# Keep Hybrid predictions within the valid rating scale
evaluation_data["hybrid_prediction"] = evaluation_data["hybrid_prediction"].clip(0.5, 5.0)


# ============================================================
# 15. PRECISION, RECALL, F1-SCORE AND COVERAGE
# ============================================================

def calculate_top_k_metrics(data, score_column, k=10, relevance_threshold=4.0):
    """
    Calculate:
    Precision@K
    Recall@K
    F1@K
    Coverage

    A movie with an actual rating of 4.0 or above
    is treated as relevant.
    """

    # Store each user's Precision and Recall
    user_precisions = []
    user_recalls = []

    # Store all unique recommended movie IDs
    recommended_movie_ids = set()

    # Evaluate one user at a time
    for user_id, user_data in data.groupby("userId"):

        # Sort movies by predicted score
        ranked_data = user_data.sort_values(by=score_column, ascending=False)

        # Select the Top K movies
        top_k_data = ranked_data.head(k)

        # Count all relevant movies for the user
        relevant_count = int((ranked_data["rating"] >= relevance_threshold).sum())

        # Count relevant movies appearing in the Top K
        relevant_recommended_count = int((top_k_data["rating"] >= relevance_threshold).sum())

        # Calculate Precision
        precision = relevant_recommended_count / len(top_k_data) if len(top_k_data) > 0 else 0.0

        # Calculate Recall
        recall = relevant_recommended_count / relevant_count if relevant_count > 0 else 0.0

        # Store user results
        user_precisions.append(precision)
        user_recalls.append(recall)

        # Add recommended movie IDs for coverage calculation
        recommended_movie_ids.update(top_k_data["movieId"].tolist())

    # Calculate average Precision
    average_precision = float(np.mean(user_precisions)) if user_precisions else 0.0

    # Calculate average Recall
    average_recall = float(np.mean(user_recalls)) if user_recalls else 0.0

    # Calculate F1-score
    average_f1 = 2 * average_precision * average_recall / (average_precision + average_recall) if average_precision + average_recall > 0 else 0.0

    # Calculate catalogue coverage
    coverage = len(recommended_movie_ids) / movie_data["movieId"].nunique()

    # Return all measurements
    return average_precision, average_recall, average_f1, coverage


# ============================================================
# 16. RMSE AND MAE EVALUATION
# ============================================================

def evaluate_method(data, method_name, score_column, k=10, relevance_threshold=4.0):
    """
    Evaluate one recommendation method using:
    RMSE
    MAE
    Precision@K
    Recall@K
    F1@K
    Coverage
    """

    # Calculate Root Mean Squared Error
    rmse = float(np.sqrt(mean_squared_error(data["rating"], data[score_column])))

    # Calculate Mean Absolute Error
    mae = float(mean_absolute_error(data["rating"], data[score_column]))

    # Calculate ranking measurements
    precision, recall, f1, coverage = calculate_top_k_metrics(data, score_column, k=k, relevance_threshold=relevance_threshold)

    # Return results as dictionary
    return {
        "Method": method_name,
        "RMSE": rmse,
        "MAE": mae,
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"F1@{k}": f1,
        "Coverage": coverage
    }


# ============================================================
# 17. EVALUATE ALL FOUR METHODS
# ============================================================

# Create an empty list for model results
results = []

# Evaluate Popularity-Based Recommendation
results.append(evaluate_method(evaluation_data, "Popularity-Based", "popularity_prediction", k=10, relevance_threshold=4.0))

# Evaluate Content-Based Filtering
results.append(evaluate_method(evaluation_data, "Content-Based", "content_prediction", k=10, relevance_threshold=4.0))

# Evaluate Collaborative Filtering
results.append(evaluate_method(evaluation_data, "Collaborative Filtering", "collaborative_prediction", k=10, relevance_threshold=4.0))

# Evaluate Hybrid Recommendation
results.append(evaluate_method(evaluation_data, "Hybrid", "hybrid_prediction", k=10, relevance_threshold=4.0))

# Convert model results into DataFrame
results_table = pd.DataFrame(results)

# Display model results
print("\nModel Evaluation Results")
print(results_table.round(4).to_string(index=False))

# Save results into CSV file
results_table.to_csv("movie_recommender_evaluation_results.csv", index=False)


# ============================================================
# 18. F1-SCORE COMPARISON CHART
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(results_table["Method"], results_table["F1@10"])
plt.title("F1@10 Comparison")
plt.xlabel("Recommendation Method")
plt.ylabel("F1@10")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ============================================================
# 19. RMSE COMPARISON CHART
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(results_table["Method"], results_table["RMSE"])
plt.title("RMSE Comparison")
plt.xlabel("Recommendation Method")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ============================================================
# 20. DEMONSTRATE ALL FOUR RECOMMENDATION METHODS
# ============================================================

# Select one example user
example_user_id = int(ratings["userId"].iloc[0])

# Select one example movie
example_movie_title = "Toy Story"

# Use another movie if Toy Story does not exist
if content_movies[content_movies["title"].str.lower() == example_movie_title.lower()].empty:
    example_movie_title = str(content_movies["title"].iloc[0])


# ============================================================
# METHOD 1 OUTPUT
# ============================================================

print("\n============================================================")
print("METHOD 1: POPULARITY-BASED RECOMMENDATIONS")
print("============================================================")

# Display Top 10 popular movies
print(recommend_by_popularity(top_n=10).to_string(index=False))


# ============================================================
# METHOD 2 OUTPUT
# ============================================================

print("\n============================================================")
print(f"METHOD 2: CONTENT-BASED RECOMMENDATIONS FOR '{example_movie_title}'")
print("============================================================")

# Display Top 10 content-similar movies
print(recommend_by_content(example_movie_title, top_n=10).to_string(index=False))


# ============================================================
# METHOD 3 OUTPUT
# ============================================================

print("\n============================================================")
print(f"METHOD 3: COLLABORATIVE RECOMMENDATIONS FOR USER {example_user_id}")
print("============================================================")

# Display Top 10 personalised Collaborative recommendations
print(recommend_by_collaborative(example_user_id, top_n=10).to_string(index=False))


# ============================================================
# METHOD 4 OUTPUT
# ============================================================

print("\n============================================================")
print(f"METHOD 4: HYBRID RECOMMENDATIONS FOR USER {example_user_id}")
print("============================================================")

# Record starting time
start_time = time.perf_counter()

# Generate Hybrid recommendations
hybrid_recommendations = recommend_by_hybrid(example_user_id, example_movie_title, top_n=10)

# Calculate response time
response_time = time.perf_counter() - start_time

# Display Hybrid recommendations
print(hybrid_recommendations.to_string(index=False))

# Display response time
print(f"\nHybrid recommendation response time: {response_time:.4f} seconds")


# ============================================================
# 21. SAVE RECOMMENDATION RESULTS
# ============================================================

# Save Popularity-Based recommendations
#recommend_by_popularity(top_n=10).to_csv("popular_movie_recommendations.csv", index=False)

# Save Content-Based recommendations
#recommend_by_content(example_movie_title, top_n=10).to_csv("content_movie_recommendations.csv", index=False)

# Save Collaborative Filtering recommendations
#recommend_by_collaborative(example_user_id, top_n=10).to_csv("collaborative_movie_recommendations.csv", index=False)

# Save Hybrid recommendations
#hybrid_recommendations.to_csv("hybrid_movie_recommendations.csv", index=False)


# ============================================================
# 22. COMPLETION MESSAGE
# ============================================================

#print("\nProgram completed successfully.")

#print("\nGenerated files:")
#print("1. movie_recommender_evaluation_results.csv")
#print("2. popular_movie_recommendations.csv")
#print("3. content_movie_recommendations.csv")
#print("4. collaborative_movie_recommendations.csv")
#print("5. hybrid_movie_recommendations.csv")