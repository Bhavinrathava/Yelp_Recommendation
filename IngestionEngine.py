import json
from pymongo import MongoClient

# MongoDB connection settings
mongo_uri = "mongodb+srv://mlproject:mlproject@bhavinmongocluster.5t6smyb.mongodb.net/Yelp?retryWrites=true&w=majority"

# Path to your JSON file
json_file = 'Data/yelp_academic_dataset_business.json'

jsonPaths = ['Data/yelp_academic_dataset_business.json','Data/yelp_academic_dataset_checkin.json','Data/yelp_academic_dataset_review.json','Data/yelp_academic_dataset_tip.json','Data/yelp_academic_dataset_user.json']

CollectionNames = ["Business","Checkin","Review","Tip","User"]

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def insert_data_to_mongodb(data, mongo_uri):
    client = MongoClient(mongo_uri)
    db = client.get_default_database()  # Replace with your desired database name

    collection = db['Business']  # Replace with your desired collection name

    for line in data:
        try:
            json_data = json.loads(line)
            # Insert the data into the MongoDB collection
            collection.insert_one(json_data)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {line}")

    client.close()

if __name__ == "__main__":
    with open(json_file, 'r') as file:
        json_data = file.readlines()
    
    insert_data_to_mongodb(json_data, mongo_uri)

    for collection,jsonPath in zip(CollectionNames, jsonPaths):
        with open(jsonPath, 'r') as file:
            json_data = file.readlines()
        insert_data_to_mongodb(json_data, mongo_uri)
        print("Data inserted into MongoDB's {} Collection successfully!".format(collection))
    print("Data inserted into MongoDB successfully!")