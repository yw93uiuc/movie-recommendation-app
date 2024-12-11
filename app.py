import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Function to compute cosine similarity
def compute_cosine_similarity(matrix):
    norm = np.sqrt(np.nansum(matrix**2, axis=0))  # Compute norms for each column
    similarity = np.dot(matrix.T.fillna(0), matrix.fillna(0)) / (norm[:, None] * norm[None, :])
    similarity[np.isnan(similarity)] = 0  # Replace NaNs with 0
    return similarity

# Function to filter top K similarities
def filter_top_k(similarity_matrix, k=30):
    filtered_matrix = np.zeros_like(similarity_matrix)
    for i in range(similarity_matrix.shape[0]):
        top_k_indices = np.argsort(-similarity_matrix[i, :])[:k]
        filtered_matrix[i, top_k_indices] = similarity_matrix[i, top_k_indices]
    return filtered_matrix

# Function for generating recommendations
def myIBCF(new_user_ratings, similarity_matrix, movie_subset, all_movies, k=10):
    # Compute weighted sum and normalization
    weighted_sum = np.nansum(similarity_matrix * new_user_ratings[:, None], axis=0)
    normalization = np.nansum(np.abs(similarity_matrix), axis=0)
    normalization[normalization == 0] = 1  # Avoid division by zero

    predicted_ratings = weighted_sum / normalization

    # Get top k recommended movies
    recommended_indices = np.argsort(-predicted_ratings)[:k]
    recommendations = pd.DataFrame({
        'MovieID': all_movies['MovieID'].iloc[recommended_indices].values,
        'Title': all_movies['Title'].iloc[recommended_indices].values
    })
    return recommendations

# Load datasets
movies_path = 'movies.dat'
ratings_path = 'ratings.dat'
image_folder = 'MovieImages'  # Path to the folder containing movie images

# Load movies and ratings data
movies_df = pd.read_csv(movies_path, sep='::', engine='python', header=None, encoding='latin-1')
movies_df.columns = ['MovieID', 'Title', 'Genres']

ratings_df = pd.read_csv(ratings_path, sep='::', engine='python', header=None, encoding='latin-1')
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Create and normalize the rating matrix
rating_matrix = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
rating_matrix_centered = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0)

# Compute similarity matrix dynamically
st.write("Computing similarity matrix, please wait...")
similarity_matrix = compute_cosine_similarity(rating_matrix_centered)

# Create a mapping of MovieID to matrix indices
movie_id_to_index = {movie_id: i for i, movie_id in enumerate(rating_matrix.columns)}

# Filter sample movies to only those in the similarity matrix
sample_movies = movies_df[movies_df['MovieID'].isin(movie_id_to_index.keys())].sample(10, random_state=42)
sample_indices = sample_movies['MovieID'].apply(lambda x: movie_id_to_index[x]).values
similarity_subset = similarity_matrix[sample_indices]

# Streamlit App
st.title("Movie Recommendation System")
st.write("Rate the following movies to get personalized recommendations!")

# Rating inputs (display movies in a 5-column, 2-row grid)
user_ratings = []
rows = 2
cols_per_row = 5
for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i * cols_per_row + j
        if idx < len(sample_movies):
            row = sample_movies.iloc[idx]
            image_path = os.path.join(image_folder, f"{row['MovieID']}.jpg")
            with cols[j]:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    st.image(img, caption=row['Title'], use_container_width=True)
                rating = st.slider(f"Rate '{row['Title']}'", 0, 5, 0, key=row['MovieID'])  # Unique key for each slider
                user_ratings.append(rating)

# Convert user ratings to numpy array
new_user_ratings = np.zeros(len(rating_matrix.columns))  # Initialize with zeros
for i, rating in enumerate(user_ratings):
    movie_id = sample_movies.iloc[i]['MovieID']
    if movie_id in movie_id_to_index:
        new_user_ratings[movie_id_to_index[movie_id]] = rating

# Generate recommendations
if st.button("Get Recommendations"):
    recommendations = myIBCF(new_user_ratings, similarity_matrix, sample_movies, movies_df)

    st.write("Top 10 Movie Recommendations:")
    rows = 2
    cols_per_row = 5
    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(recommendations):
                row = recommendations.iloc[idx]
                image_path = os.path.join(image_folder, f"{row['MovieID']}.jpg")
                with cols[j]:
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        st.image(img, caption=row['Title'], use_container_width=True)
                    else:
                        st.write(row['Title'])
