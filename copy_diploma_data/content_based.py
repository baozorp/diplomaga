import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import os

# Function to recommend similar objects


def content_based():

    user_data_path = "data/current_user.csv"
    exhibits_path = "data/object_exhibits.csv"

    if not os.path.isfile(user_data_path):
        raise FileNotFoundError(f"Current user data file {user_data_path} not found.")
    if not os.path.isfile(exhibits_path):
        raise FileNotFoundError(f"Object exhibits data file {exhibits_path} not found.")

    current_user = pd.read_csv(user_data_path, usecols=['user_id', 'object_id'])
    # Get the ID of the last object visited by user 0
    last_object_id = current_user.iloc[-1]['object_id']

    # Load the object database into a DataFrame
    df = pd.read_csv(exhibits_path, usecols=lambda column: column != 'ID')

    # Select relevant features for similarity calculations

    features = df.columns.tolist()

    # Encode categorical variables as numeric
    gf = pd.get_dummies(df, columns=features)
    # Calculate pairwise cosine similarity between all objects
    similarity = cosine_similarity(gf)
    list_of_objects = _recommend_objects(last_object_id, similarity, df)

    # Validation
    sorted_indexes = list_of_objects['ID'].tolist()  # Get sorted indexes
    gf_sorted = gf.loc[sorted_indexes]  # Sort gf by the sorted indexes
    item_vector = gf.loc[last_object_id]
    gf_sorted = gf_sorted.drop(labels=last_object_id)
    gf_split = np.array_split(gf_sorted, 5)

    similarities = []
    for split in gf_split:
        split_similarities = []
        for index, row in split.iterrows():
            split_vector = row.values
            cos_sim = np.dot(split_vector, item_vector) / (np.linalg.norm(split_vector) * np.linalg.norm(item_vector))
            split_similarities.append(cos_sim)
        average_similarity = sum(split_similarities) / len(split)
        similarities.append(average_similarity)
    validation_result = (similarities == sorted(similarities, reverse=True))
    if validation_result:
        print("Content based has been successfully validated")
        list_of_objects.to_csv(f"data/recs_content_based.csv", index=False)
    else:
        raise ValueError("Content base had a validation error")


def _recommend_objects(id, similarity, df):
    # Get the similarity scores of all other objects
    scores = list(enumerate(similarity[id]))

    # Sort by similarity using a vectorized operation
    sorted_indexes = np.argsort(similarity[id])[::-1]
    scores = np.array(scores)[sorted_indexes]

    # Create a DataFrame of the sorted scores
    result = pd.DataFrame({'ID': scores[:, 0], 'similarity': scores[:, 1]})

    return result

def generate_content_based_test_data(num_objects=100,
                                     num_features=5,
                                     num_categories=3,
                                     user_id=0,
                                     num_user_visits=25):
    np.random.seed(42)  # воспроизводимость

    # ====== object_museum_exhibits.csv ======
    data = []
    styles = ['roman', 'gothic', 'baroque', 'modernism', 'brutalism', 'hightech']
    for obj_id in range(num_objects):
        row = {
            'ID': obj_id
        }
        # Случайные категориальные признаки (например, 'color', 'type', 'material')
        for i in range(num_features):
            row[f'feature_{i}'] = f'{np.random.choice(styles)}_{np.random.randint(num_categories)}'
        data.append(row)

    exhibits_df = pd.DataFrame(data)
    exhibits_df.to_csv(os.path.join("data/object_exhibits.csv"), index=False)

    # ====== current_user.csv ======
    current_user_objects = np.random.choice(range(num_objects), size=num_user_visits, replace=False)
    current_user_data = [{
        'user_id': user_id,
        'object_id': obj_id
    } for obj_id in current_user_objects]

    current_user_df = pd.DataFrame(current_user_data)
    current_user_df.to_csv(os.path.join("data/current_user.csv"), index=False)

    print(f"✅ Content-based test data generated in: data")


# Пример использования
generate_content_based_test_data(
    num_objects=100,
    num_features=5,
    num_categories=4,
    num_user_visits=25
)
