import pickle
import pandas as pd


# Load model

similarity = pickle.load(
    open("model/movie_similarity.pkl", "rb")
)

movie_names = pickle.load(
    open("model/movie_names.pkl", "rb")
)


# Convert movie names into Series
movie_list = pd.Series(movie_names)
print(type(movie_names))
print(movie_list.head())


def recommend_movie(movie_name, num_recommendations=5):

    movie_name = movie_name.strip()


    # Search movie
    if movie_name not in movie_list.values:

        matches = movie_list[
            movie_list.str.contains(
                movie_name,
                case=False,
                regex=False
            )
        ]

        if len(matches) > 0:
            return "Similar movie names found:\n" + matches.head(5).to_string(index=False)

        return "Movie not found"


    # Get index of movie
    movie_index = movie_list[
        movie_list == movie_name
    ].index[0]


    # Get similarity scores
    distances = similarity[movie_index]


    # Sort similarity
    movie_indices = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )


    recommendations = []


    for i in movie_indices[1:num_recommendations+1]:
        recommendations.append(
            movie_list.iloc[i[0]]
        )


    return recommendations



# Test

movie_name = input("Enter movie name: ")

result = recommend_movie(movie_name)


print("\nRecommended Movies:")
print(result)


print(type(movie_names))
print(movie_list.head())

print("Similarity Shape:", similarity.shape)
print("Movie List Length:", len(movie_list))

print("\nFirst 20 Movies")
print(movie_list.head(20).to_string())