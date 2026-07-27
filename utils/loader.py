import streamlit as st
import pandas as pd
import pickle


@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings_sample.csv")

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")

    movie_rating = ratings.groupby("movieId")["rating"].mean().reset_index()
    movie_rating.rename(columns={"rating": "Average Rating"}, inplace=True)

    rating_count = ratings.groupby("movieId")["rating"].count().reset_index()
    rating_count.rename(columns={"rating": "Rating Count"}, inplace=True)

    movies = movies.merge(movie_rating, on="movieId", how="left")
    movies = movies.merge(rating_count, on="movieId", how="left")

    movies["Average Rating"] = movies["Average Rating"].fillna(0)
    movies["Rating Count"] = movies["Rating Count"].fillna(0)

    return movies, ratings


@st.cache_resource
def load_model():

    similarity = pickle.load(
        open("model/movie_similarity.pkl", "rb")
    )

    movie_names = pickle.load(
        open("model/movie_names.pkl", "rb")
    )

    return similarity, movie_names