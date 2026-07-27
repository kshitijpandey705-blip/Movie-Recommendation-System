# ==============================
# app.py (Part 1/4)
# Movie Recommendation System
# ==============================

import streamlit as st
import pickle
import os
from pathlib import Path
import plotly.express as px
import pandas as pd

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🎬 AI Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    text-align:center;
    color:#FF4B4B;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#BBBBBB;
    font-size:18px;
    margin-bottom:30px;
}

.movie-card{
    background:#1E1E1E;
    border-radius:15px;
    padding:18px;
    margin-bottom:15px;
    border:1px solid #333333;
    transition:0.3s;
}

.movie-card:hover{
    border:1px solid #FF4B4B;
    transform:scale(1.02);
}

.metric-card{
    background:#1E1E1E;
    border-radius:15px;
    padding:20px;
    text-align:center;
    border:1px solid #333;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

hr{
    border:1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Project Paths
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MOVIE_LIST_PATH = MODEL_DIR / "movie_names.pkl"
SIMILARITY_PATH = MODEL_DIR / "movie_similarity.pkl"

MOVIES_CSV = DATA_DIR / "movies.csv"

# ------------------------------
# Load Model Files
# ------------------------------
@st.cache_resource
def load_models():

    with open(MOVIE_LIST_PATH, "rb") as f:
        movie_list = pickle.load(f)

    with open(SIMILARITY_PATH, "rb") as f:
        similarity = pickle.load(f)

    return movie_list, similarity


# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv(MOVIES_CSV)


# ------------------------------
# Recommendation Function
# ------------------------------
def recommend(movie_name, similarity, movie_list):

    movie_name = movie_name.strip()

    if movie_name not in movie_list:
        return []

    index = movie_list.index(movie_name)

    distances = similarity[index]

    movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movies:
        recommendations.append(movie_list[i[0]])

    return recommendations


# ------------------------------
# Load Everything
# ------------------------------
try:

    movie_list, similarity = load_models()
    movies_df = load_dataset()

except Exception as e:

    st.error("❌ Error Loading Project Files")
    st.exception(e)
    st.stop()

# ==============================
# app.py (Part 2/4)
# Dashboard + Sidebar + Search UI
# ==============================

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/480/movie-projector.png",
        width=120
    )

    st.title("🎬 Movie Recommender")

    st.markdown("---")

    st.markdown("""
### 📌 Features

✅ Content Based Recommendation

✅ Top 5 Similar Movies

✅ Large Movie Dataset

✅ Fast Recommendation

✅ Modern Streamlit UI

✅ Interactive Dashboard
""")

    st.markdown("---")

    st.info(
        """
Developer:
Kshitij Pandey

B.Tech CSE (AI)

Movie Recommendation Project
"""
    )


# ------------------------------
# Main Heading
# ------------------------------
st.markdown(
    '<h1 class="title">🎬 AI Movie Recommendation System</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Discover movies similar to your favourite movies using Machine Learning.</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# ------------------------------
# Dashboard Metrics
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
<div class="metric-card">
<h2>🎞️</h2>
<h3>{len(movie_list):,}</h3>
<p>Total Movies</p>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
<div class="metric-card">
<h2>🤖</h2>
<h3>ML Model</h3>
<p>Content Based Filtering</p>
</div>
""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
<div class="metric-card">
<h2>⭐</h2>
<h3>Top 5</h3>
<p>Recommendations</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------
# Movie Selection
# ------------------------------
selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_list
)


# ------------------------------
# Search Button
# ------------------------------
recommend_button = st.button(
    "🔍 Recommend Movies",
    use_container_width=True
)

st.markdown("---")

# ==============================
# app.py (Part 3/4)
# Recommendation Logic + Results
# ==============================

if recommend_button:

    with st.spinner("🔎 Finding similar movies..."):

        recommended_movies = recommend(
            selected_movie,
            similarity,
            movie_list
        )

    st.success("✅ Recommendation Generated Successfully!")

    # ------------------------------
    # Selected Movie Information
    # ------------------------------
    st.subheader("🎬 Selected Movie")

    selected_data = movies_df[
        movies_df["title"] == selected_movie
    ]

    if not selected_data.empty:

        col1, col2 = st.columns([2, 3])

        with col1:

            st.markdown(f"""
<div class="movie-card">

### 🎥 {selected_movie}

</div>
""", unsafe_allow_html=True)

        with col2:

            if "genres" in movies_df.columns:
                st.write("**Genre:**", selected_data.iloc[0]["genres"])

            if "vote_average" in movies_df.columns:
                st.write(
                    "**Rating:**",
                    round(float(selected_data.iloc[0]["vote_average"]), 1)
                )

            if "release_date" in movies_df.columns:
                st.write(
                    "**Release Date:**",
                    selected_data.iloc[0]["release_date"]
                )

            if "overview" in movies_df.columns:
                overview = str(selected_data.iloc[0]["overview"])

                if len(overview) > 350:
                    overview = overview[:350] + "..."

                st.write("**Overview:**")
                st.write(overview)

    st.markdown("---")

    # ------------------------------
    # Recommended Movies
    # ------------------------------
    st.subheader("🍿 Top 5 Recommended Movies")

    if len(recommended_movies) == 0:

        st.warning("No recommendations found.")

    else:

        cols = st.columns(2)

        for i, movie in enumerate(recommended_movies):

            movie_info = movies_df[
                movies_df["title"] == movie
            ]

            with cols[i % 2]:

                st.markdown(
                    f"""
<div class="movie-card">

### {i+1}. {movie}

</div>
""",
                    unsafe_allow_html=True
                )

                if not movie_info.empty:

                    if "genres" in movies_df.columns:
                        st.caption(
                            f"🎭 Genre: {movie_info.iloc[0]['genres']}"
                        )

                    if "vote_average" in movies_df.columns:
                        st.caption(
                            f"⭐ Rating: {round(float(movie_info.iloc[0]['vote_average']),1)}"
                        )

                    if "release_date" in movies_df.columns:
                        st.caption(
                            f"📅 {movie_info.iloc[0]['release_date']}"
                        )

    st.markdown("---")

    # ==============================
# app.py (Part 4/4)
# Statistics + Dataset Preview + Footer
# ==============================

# ------------------------------
# Dataset Statistics
# ------------------------------
st.markdown("## 📊 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎬 Total Movies", len(movie_list))

with col2:
    if "genres" in movies_df.columns:
        total_genres = (
            movies_df["genres"]
            .fillna("")
            .astype(str)
            .str.split("|")
            .explode()
            .nunique()
        )
        st.metric("🎭 Unique Genres", total_genres)
    else:
        st.metric("🎭 Unique Genres", "N/A")

with col3:
    if "vote_average" in movies_df.columns:
        avg_rating = round(movies_df["vote_average"].fillna(0).mean(), 2)
        st.metric("⭐ Average Rating", avg_rating)
    else:
        st.metric("⭐ Average Rating", "N/A")


st.markdown("---")

# ------------------------------
# Dataset Preview
# ------------------------------
with st.expander("📂 Preview Dataset"):

    preview_columns = [
        col for col in [
            "title",
            "genres",
            "vote_average",
            "release_date"
        ]
        if col in movies_df.columns
    ]

    if preview_columns:
        st.dataframe(
            movies_df[preview_columns].head(20),
            use_container_width=True
        )
    else:
        st.dataframe(
            movies_df.head(20),
            use_container_width=True
        )


# ------------------------------
# Search Information
# ------------------------------
st.info(
    """
💡 **Tip:** Search for popular movies to get the best recommendations.
Examples: Avatar, Titanic, Inception, Interstellar, The Dark Knight.
"""
)

# ------------------------------
# Analytics Dashboard
# ------------------------------

st.markdown("## 📊 Movie Analytics Dashboard")

if "genres" in movies_df.columns:

    genre_df = (
        movies_df["genres"]
        .fillna("")
        .astype(str)
        .str.split("|")
        .explode()
        .value_counts()
        .head(10)
        .reset_index()
    )

    genre_df.columns = ["Genre", "Movies"]

    fig = px.bar(
        genre_df,
        x="Genre",
        y="Movies",
        title="Top 10 Movie Genres"
    )

    st.plotly_chart(fig, use_container_width=True)

if "vote_average" in movies_df.columns:

    fig2 = px.histogram(
        movies_df,
        x="vote_average",
        nbins=20,
        title="Movie Rating Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)
# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.markdown(
    """
<div class="footer">

Made with ❤️ using Python, Streamlit & Machine Learning

<br><br>

© 2026 Kshtij Pandey 

</div>
""",
    unsafe_allow_html=True,
)