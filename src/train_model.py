import pandas as pd

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv", nrows=500000)

# Merge datasets
movie_data = movies.merge(ratings, on="movieId")

print("=" * 50)
print("Merged Dataset")
print("=" * 50)

print(movie_data.head())

print("\nShape:", movie_data.shape)


movies['title'] = movies['title'].str.strip()

# Calculate average rating of each movie
movie_rating = movie_data.groupby("title")["rating"].mean().reset_index()

# Rename column
movie_rating.rename(columns={"rating": "average_rating"}, inplace=True)

# Sort movies by rating
top_movies = movie_rating.sort_values(by="average_rating", ascending=False)

print("\n")
print("=" * 50)
print("Top Rated Movies")
print("=" * 50)

print(top_movies.head(20))

# Count how many ratings each movie received
rating_count = movie_data.groupby("title")["rating"].count().reset_index()

# Rename column
rating_count.rename(columns={"rating": "rating_count"}, inplace=True)

# Merge average rating with rating count
final_rating = movie_rating.merge(rating_count, on="title")

# Keep only movies with at least 50 ratings
popular_movies = final_rating[final_rating["rating_count"] >= 50]

# Sort by average rating
popular_movies = popular_movies.sort_values(
    by="average_rating",
    ascending=False
)

print("\n")
print("=" * 50)
print("Popular Movies")
print("=" * 50)

print(popular_movies.head(20))


# -----------------------------
# Keep only popular movies
# -----------------------------

movie_rating_count = (
    movie_data.groupby("title")["rating"]
    .count()
    .reset_index()
)

movie_rating_count.rename(
    columns={"rating": "rating_count"},
    inplace=True
)

# Keep movies having at least 100 ratings
popular_movies = movie_rating_count[
    movie_rating_count["rating_count"] >= 100
]["title"]

movie_data = movie_data[
    movie_data["title"].isin(popular_movies)
]

print("Popular Movies:", movie_data["title"].nunique())

# -----------------------------
# User Movie Matrix
# -----------------------------

user_movie_matrix = movie_data.pivot_table(
    index="title",
    columns="userId",
    values="rating"
).fillna(0)

print("Matrix Shape:", user_movie_matrix.shape)

# Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

print("\n")
print("=" * 50)
print("User-Movie Matrix")
print("=" * 50)

print(user_movie_matrix.head())

print("\nMatrix Shape:", user_movie_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity


# Calculate Movie Similarity
movie_similarity = cosine_similarity(
    user_movie_matrix.astype("float32")
)


print("\n")
print("=" * 50)
print("Movie Similarity Matrix")
print("=" * 50)

print(movie_similarity[:5])


print("\nSimilarity Matrix Shape:", movie_similarity.shape)



import pickle
import os


# Create model folder
os.makedirs("model", exist_ok=True)


# Save similarity matrix
with open("model/movie_similarity.pkl", "wb") as f:
    pickle.dump(movie_similarity, f)


# Save movie names
with open("model/movie_names.pkl", "wb") as f:
    pickle.dump(
        user_movie_matrix.index.tolist(),
        f
    )


print("\nModel Saved Successfully ✅")