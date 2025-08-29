import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def collaborative_system(sources_path, results_path):

    exhibit_data_path = sources_path + "data/exhibit_data.csv"
    current_user_path = sources_path + "data/current_user.csv"

    if not os.path.isfile(exhibit_data_path):
        raise FileNotFoundError(f"Exhibit data file {exhibit_data_path} not found.")

    if not os.path.isfile(current_user_path):
        raise FileNotFoundError(f"Current user data file {current_user_path} not found.")

    # Load in the exhibit data
    exhibit_data = pd.read_csv(exhibit_data_path)
    current_user = pd.read_csv(current_user_path)

    if current_user.shape[0] < 20:
        return

    # Clip the time_spent values to 300 scores
    exhibit_data = pd.concat([exhibit_data, current_user], ignore_index=True)
    exhibit_data['time_spent'] = exhibit_data['time_spent'].clip(upper=300)

    # Create a pivot table of the exhibit data to get a matrix of user-item interactions
    user_item_matrix = exhibit_data.pivot_table(values='time_spent', index='user_id', columns='object_id').fillna(0)
    # Compute user similarity based on user-item interactions
    # Here, we are using cosine similarity as the similarity metric
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get collaborative filtering recommendations for a specific user and save the results to a CSV file
    user_id = exhibit_data['user_id'].min()

    prediction = get_collaborative_filtering_recs(user_id, user_item_matrix, user_similarity_df)

    # Validate the collaborative filtering system using a random subset of the data
    collaborative_filtering_validation(user_id, user_item_matrix, user_similarity_df, exhibit_data)

    prediction = prediction.sort_values(ascending=False)
    prediction_df = pd.DataFrame(prediction, columns=['seconds'])
    prediction_df.reset_index(inplace=True)
    prediction_df.rename(columns={'object_id': 'ID'}, inplace=True)
    prediction_df.to_csv(f"{results_path}/data/recs_collaborative.csv", index=False)


def get_collaborative_filtering_recs(user_id, user_item_matrix, user_similarity_df, start_from=1, end_with=11, validation=False):
    # Get the user's interactions
    user_interactions = user_item_matrix.loc[user_id]

    # Get the similarity scores for the user
    user_similarities = user_similarity_df[user_id]

    # Sort the recommendations by scores
    top_recs = user_similarities.sort_values(ascending=False)

    # Get the top recommendations within the specified range
    cf_recs = top_recs[start_from:end_with]

    # Create a new matrix containing only the users in cf_recs
    new_matrix = user_item_matrix.loc[cf_recs.index]

    # Multiply new_matrix by the sum of user similarities in cf_recs
    coeff = cf_recs.sum()
    new_matrix = new_matrix.mul(cf_recs, axis=0)

    # Sum the columns in new_matrix
    prediction = new_matrix.sum(axis=0)

    # Exclude items that the user has already rated
    user_interactions = user_item_matrix.loc[user_id]
    user_interactions = user_interactions[user_interactions > 0] if validation else user_interactions[user_interactions == False]
    prediction = prediction[user_interactions.index] / coeff
    return prediction


def collaborative_filtering_validation(user_id, user_item_matrix, user_similarity_df, exhibit_data):
    # Get user IDs from the exhibit_data
    user_ids = exhibit_data['user_id'].unique()

    number_of_parts = 5

    # Calculate the number of users in each part of the dataset
    part = int(len(user_ids) / number_of_parts)

    # Initialize validation_result to True
    validation_result = True

    mean_score_array = [0]
    start_from = 1 - part
    end_with = 0

    # Iterate over each part of the dataset
    for _ in range(number_of_parts):
        # Get the collaborative filtering recommendations for a sample user
        start_from += part
        end_with += part
        user_interactions = user_item_matrix.loc[user_id]
        user_interactions = user_interactions[user_interactions > 0]
        prediction = get_collaborative_filtering_recs(user_id, user_item_matrix, user_similarity_df, start_from, end_with, validation=True)

        # Calculate the mean absolute error between the predicted and actual ratings
        prediction = prediction / user_interactions
        prediction = (prediction - 1).abs()

        # Add up the results and divide by the length of prediction
        mean_score = prediction.sum() / len(prediction)

        # Check if the current mean_score is greater than the previous mean_score in the array
        validation_result = validation_result and (mean_score > max(mean_score_array))

        # Append the current mean_score to the array
        mean_score_array.append(mean_score)

    # Print the validation result
    if validation_result:
        print("Collaborative filtering has been successfully validated")
    else:
        raise ValueError("Collaborative filtering had a validation error")




def generate_collaborative_test_data(sources_path,
                                     num_users=50,
                                     num_objects=50,
                                     coeff=0.1,
                                     service_time_interval=(0, 10),
                                     uniform_interval=(0, 2)):
    np.random.seed(42)  # для воспроизводимости

    def generate_time():
        base = np.random.uniform(*service_time_interval)
        noise = np.random.uniform(*uniform_interval)
        return base * coeff + noise

    # ====== exhibit_data.csv ======
    data = []
    for user_id in range(1, num_users + 1):
        viewed_objects = np.random.choice(range(num_objects), size=np.random.randint(5, 20), replace=False)
        for obj in viewed_objects:
            time_spent = generate_time()
            data.append({
                'user_id': user_id,
                'object_id': obj,
                'time_spent': time_spent
            })

    exhibit_df = pd.DataFrame(data)
    exhibit_df.to_csv(os.path.join(sources_path, "data/exhibit_data.csv"), index=False)

    # ====== current_user.csv ======
    current_user_data = []
    current_user_objects = np.random.choice(range(num_objects), size=25, replace=False)
    for obj in current_user_objects:
        time_spent = generate_time()
        current_user_data.append({
            'user_id': 0,
            'object_id': obj,
            'time_spent': time_spent
        })

    current_user_df = pd.DataFrame(current_user_data)
    current_user_df.to_csv(os.path.join(sources_path, "data/current_user.csv"), index=False)

    print(f"✅ Test data generated in: {sources_path}")


generate_collaborative_test_data(
    sources_path="./",
    coeff=0.1,
    num_objects=50,
    service_time_interval=(0, 10),
    uniform_interval=(0, 2)
)
def generate_collaborative_test_data(sources_path,
                                     num_users=50,
                                     num_objects=50,
                                     coeff=0.1,
                                     service_time_interval=(0, 10),
                                     uniform_interval=(0, 2)):
    np.random.seed(42)  # для воспроизводимости

    def generate_time():
        base = np.random.uniform(*service_time_interval)
        noise = np.random.uniform(*uniform_interval)
        return base * coeff + noise

    # ====== exhibit_data.csv ======
    data = []
    for user_id in range(1, num_users + 1):
        viewed_objects = np.random.choice(range(num_objects), size=np.random.randint(5, 20), replace=False)
        for obj in viewed_objects:
            time_spent = generate_time()
            data.append({
                'user_id': user_id,
                'object_id': obj,
                'time_spent': time_spent
            })

    exhibit_df = pd.DataFrame(data)
    exhibit_df.to_csv(os.path.join(sources_path, "data/exhibit_data.csv"), index=False)

    # ====== current_user.csv ======
    current_user_data = []
    current_user_objects = np.random.choice(range(num_objects), size=25, replace=False)
    for obj in current_user_objects:
        time_spent = generate_time()
        current_user_data.append({
            'user_id': 0,
            'object_id': obj,
            'time_spent': time_spent
        })

    current_user_df = pd.DataFrame(current_user_data)
    current_user_df.to_csv(os.path.join(sources_path, "data/current_user.csv"), index=False)

    print(f"✅ Test data generated in: {sources_path}")


generate_collaborative_test_data(
    sources_path="./",
    coeff=0.1,
    num_objects=50,
    service_time_interval=(0, 10),
    uniform_interval=(0, 2)
)