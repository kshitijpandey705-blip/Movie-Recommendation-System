import pandas as pd

print("Loading ratings.csv...")

ratings = pd.read_csv("data/ratings.csv")

print("Creating sample...")

ratings.sample(
    n=50000,
    random_state=42
).to_csv(
    "data/ratings_sample.csv",
    index=False
)

print("Done!")