import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lit, udf, when, isnan, count, countDistinct, desc, asc, row_number, monotonically_increasing_id
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def readData():
    # import the data (chunksize returns jsonReader for iteration)
    businesses = pd.read_json("Data/yelp_academic_dataset_business.json", lines=True, orient='columns', chunksize=100000)
    reviews = pd.read_json("Data/yelp_academic_dataset_review.json", lines=True, orient='columns', chunksize=100000)
    business_chunk = None
    review_chunk = None
    # read the data
    for business in businesses:
        business_chunk = business
        break

    for review in reviews:
        review_chunk = review
        break
    return business_chunk, review_chunk

def filterBusinesses(businesses):
    business_subset = businesses[['business_id','name','address', 'categories', 'attributes','stars']]
    business_subset = business_subset[business_subset['categories'].str.contains('Restaurant.*')==True].reset_index()
    business_subset = business_subset[['business_id', 'name', 'address']]
    return business_subset

def filterReviews(reviews):
    df_review = reviews[['user_id','business_id','stars', 'date']]
    return df_review

def combineDataframes(businesses, reviews):
    all_combined = pd.merge(reviews, businesses, on='business_id')
    return all_combined

def createPivotTable(all_combined):
    rating_crosstab = all_combined.pivot_table(values='stars', index='user_id', columns='name', fill_value=0)
    return rating_crosstab


def calculate_similarity(user_item_matrix):
    """
    Calculate the cosine similarity matrix from the user-item matrix
    """
    similarity = cosine_similarity(user_item_matrix)
    np.fill_diagonal(similarity, 0)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Prediction of Ratings 

def predict_ratings(similarity, user_item_matrix, user_id):
    """
    Predict ratings for all items for a given user
    """
    total_similarity = similarity[user_id].sum()
    weighted_sum = np.dot(similarity[user_id], user_item_matrix.fillna(0))

    # Avoid division by zero
    if total_similarity == 0:
        total_similarity = 1

    predictions = weighted_sum / total_similarity
    predictions = pd.Series(predictions, index=user_item_matrix.columns)
    return predictions

def train_test_split_and_predict(data):
    """
    Split the data into train and test sets, predict ratings, and return the true and predicted ratings
    """
    train_user_item_matrix, test_user_item_matrix = train_test_split(data, test_size=0.2)

    similarity = calculate_similarity(train_user_item_matrix)
    true_ratings = []
    pred_ratings = []

    for user_id in test_user_item_matrix.index:
        true_rating = test_user_item_matrix.loc[user_id]
        pred_rating = predict_ratings(similarity, train_user_item_matrix, user_id)
        true_ratings.extend(true_rating[true_rating.notnull()])
        pred_ratings.extend(pred_rating[true_rating.notnull()])

    return true_ratings, pred_ratings

def evaluate_performance(data):
    """
    Evaluate the performance of the collaborative filtering algorithm
    """
    true_ratings, pred_ratings = train_test_split_and_predict(data)
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    return rmse, mae


def main():
    businesses, reviews = readData()
    businesses = filterBusinesses(businesses)

