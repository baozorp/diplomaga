import pandas as pd
import os


def interference_to_euristic(results_path, heuristic_coeff):

    heuristic_path = results_path + '/heuristic.csv'
    merge_recommendations_path = results_path + '/merged_recommendations.csv'

    if not os.path.isfile(heuristic_path):
        raise FileNotFoundError(f"Heuristic data file {heuristic_path} not found.")
    if not os.path.isfile(merge_recommendations_path):
        raise FileNotFoundError(f"Merged recommendations data file {merge_recommendations_path} not found.")

    # загрузка файлов
    heuristic = pd.read_csv(heuristic_path)
    recommendations_df = pd.read_csv(merge_recommendations_path)

    heuristic = heuristic[heuristic['ID'].isin(recommendations_df['ID'])].reset_index(drop=True)
    # Выдаем очки для результатов эвристики
    max_score = recommendations_df['merge_scores'].max()
    num_rows = heuristic.shape[0]

    heuristic['heuristic_scores'] = heuristic.index.map(lambda i: (num_rows - i) * max_score / num_rows) ** 2 / max_score * heuristic_coeff

    merged_df = pd.merge(heuristic, recommendations_df, on='ID')

    merged_df['Optimal_Scores'] = merged_df[['merge_scores', 'heuristic_scores']].max(axis=1)

    merged_df = merged_df.drop(columns={'merge_scores', 'heuristic_scores'})

    sorted_df = merged_df.sort_values(by='Optimal_Scores', ascending=False)

    sorted_df.to_csv(results_path + 'final_results.csv', index=False)

    print("Successfully interference")
