import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def readDataset(filename):
    return pd.read_csv(filename)    

def calculateItemMatrix(dataset, uID, bID, ratings):
    user_item_matrix = dataset.pivot(index=uID, columns=bID, values=ratings)
    user_item_matrix = user_item_matrix.fillna(0)
    return user_item_matrix

def calculateSimilarity(user_item_matrix):
    similarity_matrix = cosine_similarity(user_item_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    similarity_matrix = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    return similarity_matrix


def predict_ratings(similarity_matrix, user_item_matrix, user_id):
    """
    Predict ratings for all items for a given user
    """
    user_ratings = user_item_matrix.loc[user_id]
    total_similarity = similarity_matrix[user_id].sum()
    weighted_sum = np.dot(similarity_matrix[user_id], user_item_matrix.fillna(0))

    # Avoid division by zero
    if total_similarity == 0:
        total_similarity = 1

    predictions = weighted_sum / total_similarity
    predictions = pd.Series(predictions, index=user_item_matrix.columns)
    return predictions

def train_test_split_and_predict(data, bID, uID, ratings):
    """
    Split the data into train and test sets, predict ratings, and return the true and predicted ratings
    """
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_user_item_matrix = calculateItemMatrix(train_data, uID, bID, ratings)
    test_user_item_matrix = calculateItemMatrix(test_data, uID, bID, ratings)

    similarity = calculateSimilarity(train_user_item_matrix)
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
    filename = 'Data/merged_2K_data.csv'
    bID = "business_id"
    uID = "user_id"
    ratings = "stars_x"

    dataset = readDataset(filename)

    # Create a user-item matrix
    user_item_matrix = calculateItemMatrix(dataset, uID, bID, ratings)

    # Calculate user-user similarity (cosine similarity)
    similarity_matrix = cosine_similarity(user_item_matrix)

    return dataset

if __name__ == '__main__':
    main()