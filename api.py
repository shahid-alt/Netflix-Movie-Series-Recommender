"""
Flask API for Movie and Series Recommendation.
"""

from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """
    Index route of the recommendation system API.

    Returns:
        JSON: A welcome message.
    """
    return jsonify({'message': 'Welcome to the recommendation system API'}), 200

@app.route('/recommendation/<string:title>', methods=['GET'])
def recommendation(title: str):
    """
    Endpoint to get recommendations for a given title.

    Args:
        title (str): The title for which recommendations are requested.

    Returns:
        JSON: Recommendations for the given title.
    """
    recommendation_list = get_recommendation(title)
    if recommendation_list:
        return jsonify({'recommendations': recommendation_list}), 200
    return jsonify({'message': 'Title not found or no recommendations available'}), 404

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded data.
    """
    loaded_data = pd.read_csv(file_path)
    loaded_data = loaded_data.replace(np.nan, '')
    df = loaded_data[
        [
            'show_id', 'type', 'title', 'director', 'cast', 'country', 'rating', 'description'
        ]
    ]
    return df

def get_recommendation(title: str):
    """
    Get recommendations for a given title.

    Args:
        title (str): The title for which recommendations are requested.

    Returns:
        list: List of recommended titles.
    """
    recommendation_list = []
    if title in new_df['Title'].values:
        title_index = new_df[new_df['Title'] == title].index[0]
        distance = sorted(
            list(
                enumerate(
                    similarity[title_index]
                    )
                ), reverse=True, key=lambda x: x[1])
        for i in distance[1:11]:
            recommendation_list.append(new_df.iloc[i[0]].Title)
    return recommendation_list

# Load data and precompute similarity matrix
data_frame = load_data('netflix_titles.csv')
new_df = pd.read_csv('dataset.csv')

tfidf = TfidfVectorizer(
    strip_accents='ascii', analyzer='word', stop_words='english', max_features=15000
    )
vectorizer = tfidf.fit_transform(new_df['Info'])
similarity = cosine_similarity(vectorizer)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
