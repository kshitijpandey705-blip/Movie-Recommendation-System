import plotly.express as px
import streamlit as st


def show_genre_chart(movies):

    genre_data = (
        movies["genres"]
        .str.split("|")
        .explode()
        .value_counts()
        .head(10)
    )

    fig = px.bar(
        x=genre_data.index,
        y=genre_data.values,
        title="🎭 Top 10 Movie Genres",
        labels={
            "x": "Genre",
            "y": "Movies"
        },
        color=genre_data.values,
        color_continuous_scale="viridis"
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)


def show_rating_chart(movies):

    fig = px.histogram(
        movies,
        x="Average Rating",
        nbins=20,
        title="⭐ Rating Distribution",
        color_discrete_sequence=["#00CC96"]
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)