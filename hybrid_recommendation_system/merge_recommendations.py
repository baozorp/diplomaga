import pandas as pd
import os


def merge_recommendations(content_based_coeff, collaborative_coeff):
    df1_path = 'data/recs_collaborative.csv'
    df2_path =  'data/recs_content_based.csv'

    if not os.path.isfile(df2_path):
        raise FileNotFoundError(f"Recommendated content based data file {df2_path} not found.")

    df2 = pd.read_csv(df2_path)

    similarity_top = df2["similarity"].max()
    df2["score"] = 100 * df2["similarity"] / similarity_top * content_based_coeff


    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
        seconds_top = df1["seconds"].max()
        df1["score"] = 100 * df1["seconds"] / seconds_top * collaborative_coeff
    else:
        df1 = pd.DataFrame(columns=["ID", "score"])
        missing_ids = set(df2["ID"]) - set(df1["ID"])
        missing_df1 = pd.DataFrame({"ID": list(missing_ids), "score": 0})
        df1 = pd.concat([missing_df1, df1], ignore_index=True)
        current_user = pd.read_csv('data/current_user.csv')
        mask = ~df1["ID"].isin(current_user["object_id"].values)
        df1 = df1[mask]

    df = pd.merge(df1[["ID", "score"]], df2[["ID", "score"]], on="ID", suffixes=("_1", "_2"))
    df["score"] = df["score_1"] + df["score_2"]
    df = df.drop(columns=['score_1', 'score_2'])
    df = df.rename(columns={'score': 'merge_scores'})
    df = df.sort_values(by="merge_scores", ascending=False)
    df.to_csv('./data/merged_recommendations.csv', index=False)
    print("Successfully merged")
