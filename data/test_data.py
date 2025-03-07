import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from queriesPaths import getMicrobenchmarkFilePathsV104vsV105, getMicrobenchmarkFilePathsV105vsV106, getMicrobenchmarkFilePathsV106vsV107, getMicrobenchmarkFilePathsV107vsV108, getMicrobenchmarkFilePathsV108vsV109, getApplicationBenchmarkFilePathsV104vsV105, getApplicationBenchmarkFilePathsV105vsV106, getApplicationBenchmarkFilePathsV106vsV107, getApplicationBenchmarkFilePathsV107vsV108, getApplicationBenchmarkFilePathsV108vsV109, getQueriesCSVFilePaths

def test_feature_selection(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Ridge Regression on all features
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    full_rmse = mean_squared_error(y_test, ridge.predict(X_test))
    full_r2 = r2_score(y_test, ridge.predict(X_test))

    # Define thresholds to test
    thresholds = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    results = []

    for threshold in thresholds:
    # Select features with coefficients above threshold
        coefs = np.abs(ridge.coef_)
       
        selected_features = X_train.columns[coefs > threshold]
        print(f"Selected features for threshold: {threshold}")
        if len(selected_features) == 0:
            print(f"No features left for threshold: {threshold}")
            continue  # Skip if no features are left
    
        # Train Ridge on reduced features
        X_train_reduced = X_train[selected_features]
        X_test_reduced = X_test[selected_features]

        scaler = StandardScaler()
        X_train_reduced = scaler.fit_transform(X_train_reduced)
        X_test_reduced = scaler.transform(X_test_reduced)
        ridge_reduced = Ridge(alpha=threshold)
        ridge_reduced.fit(X_train_reduced, y_train)
    
        # Evaluate model
        y_pred = ridge_reduced.predict(X_test_reduced)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Threshold: {threshold}, Features: {len(selected_features)}, RMSE: {rmse}, R²: {r2}")
        results.append((threshold, len(selected_features), rmse, r2))

    # Convert results to a readable format and save to CSV
    print("Threshold | Features | RMSE | R²")
    results_df = pd.DataFrame(results, columns=['Threshold', 'Features', 'RMSE', 'R2'])
    results_df.to_csv('feature_selection_results.csv', index=False)
    
    for threshold, num_features, rmse, r2 in results:
        print(f"{threshold:.10f} | {num_features} | {rmse:.4f} | {r2:.4f} |")
        

def aggregate_data(df):
     return df.groupby(['test_name', 'version']).agg(
        mean_ms_op=('ms_op', 'mean'),
        median_ms_op=('ms_op', 'median'),
        std_ms_op=('ms_op', 'std')
    ).reset_index()


def load_and_clean_data(csv_file_paths):
    # Load and combine data from multiple CSV files
    columns = ['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op']
    dfs = []
    for file_path in csv_file_paths:
        # newPath = "../setup/"
        df = pd.read_csv(file_path, sep=';', names=columns)
        # Replace triple quotes with an empty string in the entire DataFrame
        df = df.applymap(lambda x: x.replace('"""', '').replace('"', '') if isinstance(x, str) else x)
        # df['ms_op'] = df['ms_op'] / 1_000_000
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def parseLogsFromApplicationBenchmark(log_file_paths, query_log_file_paths):
    app_benchmark_summaries = []
    
    for log_file_path in log_file_paths:
        summary = {}
        path_to_repace = log_file_path
        df = pd.read_csv(log_file_path, sep=' ', header=None)
        insert_average = df.iloc[:, 1].mean()

        if log_file_path.endswith('other.csv'):
            query_path = path_to_repace.replace('latenciesInserts_other.csv', 'latenciesQueries_other.csv')
            query_df = pd.read_csv(query_path, sep=' ', header=None)
            query_average = query_df.iloc[:, 1].mean()
            summary['mean_latency'] = insert_average + query_average
            summary['version'] = 1
        else:
            query_path = path_to_repace.replace('latenciesInserts_latest.csv', 'latenciesQueries_latest.csv')
            query_df = pd.read_csv(query_path, sep=' ', header=None)
            query_average = query_df.iloc[:, 1].mean()
            summary['mean_latency'] = insert_average + query_average
            summary['version'] = 2

        app_benchmark_summaries.append(summary)

    return pd.DataFrame(app_benchmark_summaries)

def main():
    csv_files_v1 = getMicrobenchmarkFilePathsV104vsV105()
    csv_files_v2 = getMicrobenchmarkFilePathsV105vsV106()
    csv_files_v3 = getMicrobenchmarkFilePathsV106vsV107()
    csv_files_v4 = getMicrobenchmarkFilePathsV107vsV108()
    csv_files_v5 = getMicrobenchmarkFilePathsV108vsV109()

    df_v1 = load_and_clean_data(csv_files_v1)
    df_v2 = load_and_clean_data(csv_files_v2)
    df_v3 = load_and_clean_data(csv_files_v3)
    df_v4 = load_and_clean_data(csv_files_v4)
    df_v5 = load_and_clean_data(csv_files_v5)
    df_v1 = aggregate_data(df_v1).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v2 = aggregate_data(df_v2).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v3 = aggregate_data(df_v3).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v4 = aggregate_data(df_v4).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v5 = aggregate_data(df_v5).pivot_table(index="version", columns='test_name', values='mean_ms_op')
   # Create a DataFrame with the 'version' column
   

    # Concatenate the version DataFrame with the original DataFrame
    df_v1["version_sut"] = "v104-v105"
    df_v2["version_sut"] = "v105-v106"
    df_v3["version_sut"] = "v106-v107"
    df_v4["version_sut"] = "v107-v108"
    df_v5["version_sut"] = "v108-v109"


    df_v1 = df_v1.dropna(axis=1)
    df_v2 = df_v2.dropna(axis=1)
    df_v3 = df_v3.dropna(axis=1)
    df_v4 = df_v4.dropna(axis=1)
    df_v5 = df_v5.dropna(axis=1)
    nan_indexes = df_v1[df_v1.isna().any(axis=1)].index
    

    benchmark_df_v1 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV104vsV105(), getQueriesCSVFilePaths())
    benchmark_df_v2 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV105vsV106(), getQueriesCSVFilePaths())
    benchmark_df_v3 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV106vsV107(), getQueriesCSVFilePaths())
    benchmark_df_v4 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV107vsV108(), getQueriesCSVFilePaths())
    benchmark_df_v5 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV108vsV109(), getQueriesCSVFilePaths())

    benchmark_df_v1["version_sut"] = "v104-v105"
    benchmark_df_v2["version_sut"] = "v105-v106"
    benchmark_df_v3["version_sut"] = "v106-v107"
    benchmark_df_v4["version_sut"] = "v107-v108"
    benchmark_df_v5["version_sut"] = "v108-v109"
    

    merged_df_v1 = benchmark_df_v1.merge(df_v1, on=["version", "version_sut"])
    merged_df_v2 = benchmark_df_v2.merge(df_v2, on=["version", "version_sut"])
    merged_df_v3 = benchmark_df_v3.merge(df_v3, on=["version", "version_sut"])
    merged_df_v4 = benchmark_df_v4.merge(df_v4, on=["version", "version_sut"])
    merged_df_v5 = benchmark_df_v5.merge(df_v5, on=["version", "version_sut"])

    merged_df_v1 = pd.concat([merged_df_v1, merged_df_v2, merged_df_v3, merged_df_v4, merged_df_v5], ignore_index=True)
    nan_indexes = merged_df_v1[merged_df_v1.isna().any(axis=1)].index

    merged_df_v1 = merged_df_v1.fillna(0)
    new_final_df = merged_df_v1.drop(columns=["version", "version_sut", "-memory.allowedPercent=60", "mean_latency"])
    test_feature_selection(new_final_df, merged_df_v1["mean_latency"])

if __name__ == "__main__":
    main()
