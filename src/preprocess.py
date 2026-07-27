import pandas as pd

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv", nrows=500000)

# Display first 5 rows
print("=" * 50)
print("Movies Dataset")
print("=" * 50)
print(movies.head())

print("\nShape of Movies Dataset:", movies.shape)

print("\n" + "=" * 50)
print("Ratings Dataset")
print("=" * 50)
print(ratings.head())

print("\nShape of Ratings Dataset:", ratings.shape)


print("\nMovies Columns:")
print(movies.columns)

print("\nRatings Columns:")
print(ratings.columns)

print("\nMissing Values in Movies:")
print(movies.isnull().sum())

print("\nMissing Values in Ratings:")
print(ratings.isnull().sum())

