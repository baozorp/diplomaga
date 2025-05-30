import pandas as pd
import json


def get_points_for_map():
    heuristic_ids = pd.read_csv("data/final_result.csv")["ID"].tolist()
    coords_df = pd.read_csv("data/coordinates.csv", names=["lat", "lon"])
    coords_df["ID"] = coords_df.index
    names_df = pd.read_csv("data/names.csv")
    names_df["ID"] = names_df["ID"].astype(int)
    merged = pd.merge(coords_df, names_df, on="ID")

    filtered = merged[merged["ID"].isin(heuristic_ids)]

    points = [
        {
            "coords": [row["lat"], row["lon"]],
            "name": row["name"]
        }
        for _, row in filtered.iterrows()
    ]

    with open("data/points.json", "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)
