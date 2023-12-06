import pandas as pd
import numpy as np
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

        if count == 10000:
            break

    return true_ratings, pred_ratings

def evaluate_performance(data):
    """
    Evaluate the performance of the collaborative filtering algorithm
    """
    true_ratings, pred_ratings = train_test_split_and_predict(data)
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
    all_combined = combineDataframes(businesses, reviews)
    rating_crosstab = createPivotTable(all_combined)
    return evaluate_performance(rating_crosstab)
def main():
    rmse = []
    mae = []

    for chunksize in range(1000, 20000,1000):

        rmse_, mae_ = trainCollabFiltering(chunksize)
        rmse.append(rmse_)
        mae.append(mae_)        
        print("Chunksize: {} MAE : {} RMSE : {}", chunksize, mae_, rmse_)
    
    plt.figure(figsize=(10, 6))  # Set the size of the plot
    plt.plot(range(1000, 20000,1000), rmse, label='RMSE Loss', color='blue', marker='o')  # Plot the first loss array
    plt.plot(range(1000, 20000,1000), mae, label='MAE Loss', color='red', marker='x')  # Plot the second loss array

    plt.xlabel('ChunkSize')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.title('Losses vs Chunksize')  # Title of the plot
    plt.legend()  # Display the legend

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

