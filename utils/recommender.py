import pandas as pd


def recommend(movie_name, similarity, movie_names):

    # Convert movie names to Series
    movie_list = pd.Series(movie_names).astype(str)

    # Remove extra spaces
    movie_name = movie_name.strip()

    # Exact match
    matched = movie_list[movie_list.str.strip() == movie_name]

    # If exact match not found
    if matched.empty:

        # Try partial match
        similar = movie_list[
            movie_list.str.contains(
                movie_name,
                case=False,
                regex=False,
                na=False
            )
        ]

        if similar.empty:
            return []

        movie_index = similar.index[0]

    else:
        movie_index = matched.index[0]

    # Similarity scores
    distances = similarity[movie_index]

    # Sort by similarity
    movie_indices = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_movies = []

    for i in movie_indices:
        recommended_movies.append(movie_list.iloc[i[0]])

    return recommended_movies