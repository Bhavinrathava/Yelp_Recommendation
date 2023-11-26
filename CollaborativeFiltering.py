
import pandas as pd
import numpy as np


def readDataset():
    pass

def calculateUserMatrix():
    pass

def createUserMatrix(dataset):
        # Create a user-item matrix    
        user_item_matrix = dataset.pivot(index='user_id', columns='business_id', values='stars')

        # Fill missing values with zeros (assuming no interaction means a rating of 0)
        user_item_matrix = user_item_matrix.fillna(0)

        return user_item_matrix

def UserSimilarity(user_item_matrix):
    # Calculate user-user similarity (cosine similarity)
    user_similarity = np.dot(user_item_matrix, user_item_matrix.T)
    user_norms = np.linalg.norm(user_item_matrix, axis=1)
    user_norms = user_norms[:, np.newaxis]
    user_similarity /= np.dot(user_norms, user_norms.T)
    return user_similarity

def calculateSimilarity(dataset):

    user_item_matrix = createUserMatrix(dataset)
    user_similarity = UserSimilarity(user_item_matrix)







