import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, Reshape, Dot, Add, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

def readData(chunksize=20000):
    # import the data (chunksize returns jsonReader for iteration)
    businesses = pd.read_json("Data/yelp_academic_dataset_business.json", lines=True, orient='columns', chunksize=chunksize)
    reviews = pd.read_json("Data/yelp_academic_dataset_review.json", lines=True, orient='columns', chunksize=chunksize)
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

def predict_ratings(similarity, userItemMatrix, userID):
    """
    Predict ratings for all items for a given user
    """
    sumOfSimilarity = similarity[userID].sum()
    weightedRatings = np.dot(similarity[userID], userItemMatrix.fillna(0))

    # Avoid division by zero
    if sumOfSimilarity == 0:
        sumOfSimilarity = 1

    predictions = weightedRatings / sumOfSimilarity
    predictions = pd.Series(predictions, index=userItemMatrix.columns)
    return predictions

def consimeSimilarity(vectorA, vectorB):
    """ Cosine Similarity Between 2 Vectors : (A.B) / (||A|| * ||B||)"""
    # Taking the dot Product Between the given Vectors
    dotProduct = np.dot(vectorA, vectorB)

    # Calculating the Norm of Vector A and Vector B
    normOfA = np.linalg.norm(vectorA)
    normOfB = np.linalg.norm(vectorB)

    # Calculating the Cosine Similarity
    return dotProduct / (normOfA * normOfB)

def compute_cosine_similarity_matrix(dataset):
    # Taking the shape of the dataset
    numUsers = dataset.shape[0]

    # Creating a matrix of zeros
    similarityMatrix = np.zeros((numUsers, numUsers))

    # Calculating the similarity between users
    for i in range(numUsers):
        for j in range(i, numUsers):
            similarity = consimeSimilarity(dataset.iloc[i], dataset.iloc[j])
            similarityMatrix[i, j] = similarity
            similarityMatrix[j, i] = similarity

    # Filling the diagonal with zeros
    for i in range(numUsers):
        similarityMatrix[i, i] = 0

    # Converting the matrix to a DataFrame
    similarityMatrix = pd.DataFrame(similarityMatrix, index=dataset.index, columns=dataset.index)
    return similarityMatrix

def train_test_split_and_predict(data):
    """
    Split the data into train and test sets, predict ratings, and return the true and predicted ratings
    """
    train_user_item_matrix, test_user_item_matrix = train_test_split(data, test_size=0.8)

    similarity = calculate_similarity(data)
    print("Calculated Similarity Matrix")
    true_ratings = []
    pred_ratings = []
    count = 0
    for user_id in test_user_item_matrix.index:
        #print(count)
        count += 1
        true_rating = test_user_item_matrix.loc[user_id]
        pred_rating = predict_ratings(similarity, data, user_id)
        true_ratings.extend(true_rating[true_rating.notnull()])
        pred_ratings.extend(pred_rating[true_rating.notnull()])

        # find top k restaurants for user_id
        #k = 3
        #top_k_restaurants = pred_rating.sort_values(ascending=False).index.values[:k]
        # print the names of top k restaurants
        #print(top_k_restaurants)
        random_user = random.sample(list(test_user_item_matrix.index), 1)
        if user_id == random_user[0]:
            print("Random User: ", user_id)
            # Recommend top 3 restaurants to the random user
            top_k_restaurants = pred_rating.sort_values(ascending=False).index.values[:3]

            # print predicted rating for top 3 restaurants
            print("Predicted Ratings: ", pred_rating[top_k_restaurants])
            print("Recommended Restaurants: ", top_k_restaurants)

            # Actual top 3 restaturants rated by the random user
            actual_top_k_restaurants = true_rating.sort_values(ascending=False).index.values[:3]
            print("Actual Ratings: ", true_rating[actual_top_k_restaurants])
            print("Actual Restaurants: ", actual_top_k_restaurants)


    return true_ratings, pred_ratings

def evaluate_performance(data):
    """
    Evaluate the performance of the collaborative filtering algorithm
    """
    true_ratings, pred_ratings = train_test_split_and_predict(data)

    # # Create a mask for non-zero true values
    # non_zero_mask = true_ratings != 0
    # print("Non Zero Mask Created")
    # print(non_zero_mask)
    # # Filter both arrays using the mask
    # true_ratings = true_ratings[non_zero_mask]
    # pred_ratings = pred_ratings[non_zero_mask]

    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    return rmse, mae

class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        
        return x
    
def Recommender(n_users, n_rests, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    ub = EmbeddingLayer(n_users, 1)(user)
    
    restaurant = Input(shape=(1,))
    m = EmbeddingLayer(n_rests, n_factors)(restaurant)
    mb = EmbeddingLayer(n_rests, 1)(restaurant)   
    
    x = Dot(axes=1)([u, m])
    x = Add()([x, ub, mb])
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)  
    
    model = Model(inputs=[user, restaurant], outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)  
    
    return model

def encodingData(dataset):
    user_encode = LabelEncoder()
    dataset['user'] = user_encode.fit_transform(dataset['user_id'].values)
    n_users = dataset['user'].nunique()

    item_encode = LabelEncoder()

    dataset['business'] = item_encode.fit_transform(dataset['business_id'].values)
    n_rests = dataset['business'].nunique()

    dataset['stars'] = dataset['stars'].values#.astype(np.float32)

    min_rating = min(dataset['stars'])
    max_rating = max(dataset['stars'])

    print(n_users, n_rests, min_rating, max_rating)

    return dataset

def trainNN(combinedDataset):
    encodedDataset = encodingData(combinedDataset)

    X = encodedDataset[['user', 'business']].values
    y = encodedDataset['stars'].values

    X_train_keras, X_test_keras, y_train_keras, y_test_keras = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_keras.shape, X_test_keras.shape, y_train_keras.shape, y_test_keras.shape

    
    X_train_array = [X_train_keras[:, 0], X_train_keras[:, 1]]
    X_test_array = [X_test_keras[:, 0], X_test_keras[:, 1]]


    n_factors = 8
    n_users = combinedDataset['user'].nunique()
    n_rests = combinedDataset['business'].nunique()
    min_rating = min(combinedDataset['stars'])
    max_rating = max(combinedDataset['stars'])
    
    keras_model = Recommender(n_users, n_rests, n_factors, min_rating, max_rating)
    keras_model.summary()

    keras_model.fit(x=X_train_array, y=y_train_keras, batch_size=64,\
                          epochs=10, verbose=1, validation_data=(X_test_array, y_test_keras))
    predictions = keras_model.predict(X_test_array)

    # create the df_test table with prediction results
    df_test = pd.DataFrame(X_test_keras[:,0])
    df_test.rename(columns={0: "user"}, inplace=True)
    df_test['business'] = X_test_keras[:,1]
    df_test['stars'] = y_test_keras
    df_test["predictions"] = predictions
    df_test.head()

    rmse = np.sqrt(mean_squared_error(df_test["stars"], df_test["predictions"]))
    mae = mean_absolute_error(df_test["stars"], df_test["predictions"])
    return rmse, mae

def trainCollabFiltering(chunksize):
    businesses, reviews = readData(chunksize)
    businesses = filterBusinesses(businesses)
    reviews = filterReviews(reviews)
    combinedDataset = combineDataframes(businesses, reviews)
    userItemMatrix = createPivotTable(combinedDataset)
    return evaluate_performance(userItemMatrix)


def main():
    rmse = []
    mae = []
    chuckrange = range(1000, 15000,1000)
    for chunksize in chuckrange:

        rmse_, mae_ = trainCollabFiltering(chunksize)
        rmse.append(rmse_)
        mae.append(mae_)        
        print(f"Chunksize: {chunksize} MAE : {mae_} RMSE : {rmse_}")
    
    plt.figure(figsize=(10, 6)) 
    plt.plot(chuckrange, rmse, label='RMSE Loss', color='blue', marker='o')  
    plt.plot(chuckrange, mae, label='MAE Loss', color='red', marker='x')  

    plt.xlabel('ChunkSize') 
    plt.ylabel('Loss') 
    plt.title('Losses vs Chunksize')  
    plt.legend() 

    plt.show() 

    
    businesses, reviews = readData(50000)
    businesses = filterBusinesses(businesses)
    reviews = filterReviews(reviews)
    all_combined = combineDataframes(businesses, reviews)
    ann_rmse, ann_mae = trainNN(all_combined)

    print("ANN - RMSE: ", ann_rmse)
    print("ANN - MAE: ", ann_mae)

if __name__ == "__main__":
    main()

