import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.linear_model import Ridge
from queriesPaths import getFilePaths, getPredictionFilePaths, getQueriesCSVFilePaths, getInsertsCSVFilePaths, getMicrobenchmarkFilePathsV104vsV105, getMicrobenchmarkFilePathsV105vsV106, getMicrobenchmarkFilePathsV106vsV107, getApplicationBenchmarkFilePathsV104vsV105, getApplicationBenchmarkFilePathsV105vsV106, getApplicationBenchmarkFilePathsV106vsV107
from parse_queries_log import parse_queries_log


def extract_summary_metrics(log_file_path):
    # Read the log file
    with open(log_file_path, 'r') as file:
        content = file.read()
    
    # Use regex to extract the summary metrics
    mean_rate_match = re.search(r'mean rate (\d+\.\d+) metrics/sec', content)
    mean_rate = float(mean_rate_match.group(1)) if mean_rate_match else None

    query_rate_match = re.search(r'Overall query rate (\d+\.\d+) queries/sec', content)
    query_rate = float(query_rate_match.group(1)) if query_rate_match else None

    latency_match = re.search(r'min:\s+(\d+\.\d+)ms, med:\s+(\d+\.\d+)ms, mean:\s+(\d+\.\d+)ms, max:\s+(\d+\.\d+)ms, stddev:\s+(\d+\.\d+)ms, sum:\s+(\d+\.\d+)sec', content)
    if latency_match:
        min_latency, med_latency, mean_latency, max_latency, stddev_latency, total_latency_time = map(float, latency_match.groups())
    else:
        min_latency = med_latency = mean_latency = max_latency = stddev_latency = total_latency_time = None

    return {
        'mean_metric_rate': mean_rate,
        'mean_query_rate': query_rate,
        'min_latency': min_latency,
        'med_latency': med_latency,
        'mean_latency': mean_latency,
        'max_latency': max_latency,
        'stddev_latency': stddev_latency,
        'total_latency_time': total_latency_time
    }


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

def remove_highly_correlated_features(X, threshold=0.8):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop)

def visualize_data(X, Y):
    # Pair plot
    sns.pairplot(pd.concat([X, Y], axis=1))
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, fmt=".2f")
    plt.show()


def explore_data(X):
    print("Data Head:")
    print(X.head())
    print("\nData Description:")
    print(X.describe())
    print("\nCorrelation Matrix:")
    print(X.corr())

def apply_ridge_regression_to_nonSeenData(ridge, scaler, trained_columns, mse):
    csv_file_paths, log_file_paths, query_log_file_paths = getPredictionFilePaths()

    df = load_and_clean_data(csv_file_paths)
    aggregated_data = df.groupby(['test_name', 'version']).agg(
        mean_ms_op=('ms_op', 'mean'),
        median_ms_op=('ms_op', 'median'),
        std_ms_op=('ms_op', 'std')
    ).reset_index()

    pivoted_data = aggregated_data.pivot(index='version', columns='test_name', values='mean_ms_op').reset_index()
    pivoted_data.to_csv('output_piv2.csv', index=False)

    final_data = pivoted_data
    # Ensure the DataFrame is not empty
    if final_data.empty:
        print("The DataFrame is empty.")
        return

    missing_cols = set(trained_columns) - set(final_data.columns)

    # 2) For each missing column, create it in new_data with zeros
    # for col in missing_cols:
    #     final_data[col] = 0

    # 3) Subset & reorder columns to match exactly what the model expects
    reordered_data = final_data[trained_columns].copy()

    new_data_scaled = scaler.transform(reordered_data)

    y_new_pred = ridge.predict(new_data_scaled)

    print(f"Prediction: {y_new_pred}")
    print(f"Difference: {y_new_pred[0] - y_new_pred[1]}")
    print(f"Path: {log_file_paths}")
    app_benchmark_df = parseLogsFromApplicationBenchmark(log_file_paths, query_log_file_paths)
    print(f"Real: {app_benchmark_df['mean_latency']}")
    
    print(f"Diff to real: {y_new_pred[0] - app_benchmark_df['mean_latency'][0]}")
    print(f"Diff to real: {y_new_pred[1] - app_benchmark_df['mean_latency'][1]}")
    print(f"prediction plus mse: {y_new_pred[0] + mse}")
    print(f"prediction minus mse: {y_new_pred[0] - mse}")
    print(f"prediction plus mse: {y_new_pred[1] + mse}")
    print(f"prediction minus mse: {y_new_pred[1] - mse}")   


def apply_ridge_regression_to_all(cleaned_df):
    # Ensure the DataFrame is not empty
    if cleaned_df.empty:
        print("The DataFrame is empty.")
        return

    # Define features (X) and target (y)
    # Assuming 'mean_latency' is the target variable and all other numeric columns are features
    X = remove_perfectly_collinear_features(cleaned_df)
    y = cleaned_df['mean_latency']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained_columns = X_train.columns
    # Train the Ridge Regression model
    ridge = Ridge(alpha=4)
    ridge.fit(X_train_scaled, y_train)

    # Predict and evaluate the model
    y_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {(mse)}")
    print(f"R-squared: {r2}")
    print(f"Prediction: {y_pred.mean()}")

    return ridge, scaler, trained_columns, mse

def remove_perfectly_collinear_features(dataframe):
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr().abs()

    # Identify features with perfect collinearity
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] == 1)]

    print(len(to_drop))
    # Drop one of each pair of perfectly collinear features
    reduced_df = dataframe.drop(columns=to_drop)
    print(len(reduced_df))
    return reduced_df

def plot_grouped_heatmaps(dataframe, group_size=10):
    # Calculate the correlation matrix
    # Print original size
    print(f"\nOriginal dataframe shape: {dataframe.shape}")    
    # Print size after removing collinear features
    reduced_df = remove_perfectly_collinear_features(dataframe)
    print(f"Shape after removing collinear features: {reduced_df.shape}")
    print(f"Number of features removed: {dataframe.shape[1] - reduced_df.shape[1]}")
    corr_matrix = reduced_df.corr()

    # Get the list of features
    features = corr_matrix.columns

    # Split features into groups
    num_features = len(features)
    num_groups = int(np.ceil(num_features / group_size))

def aggregate_data(df):
     return df.groupby(['test_name', 'version']).agg(
        mean_ms_op=('ms_op', 'mean'),
        median_ms_op=('ms_op', 'median'),
        std_ms_op=('ms_op', 'std')
    ).reset_index()

def apply_ridge_to_new_data(ridge, scaler, trained_columns, mse):
    csv_files_v1, log_files_v1, query_files_v1 = getPredictionFilePaths()

    df_v1 = load_and_clean_data(csv_files_v1)
    df_v1 = aggregate_data(df_v1).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v1 = df_v1.fillna(0)
    df_v1.to_csv('predicted_data.csv', index=False)
    missing_cols = list(set(trained_columns) - set(df_v1.columns))

    missing_df = pd.DataFrame(0, index=df_v1.index, columns=missing_cols)
    
    # Concatenate the original DataFrame with the missing columns DataFrame
    df_v1 = pd.concat([df_v1, missing_df], axis=1)
    
    # Reorder columns to match the trained model's order
    df_v1 = df_v1[trained_columns]

    new_data_scaled = scaler.transform(df_v1)

    y_new_pred = ridge.predict(new_data_scaled) 

    print(f"Prediction: {y_new_pred}")
    print(f"Path: {csv_files_v1}")
    app_benchmark_df = parseLogsFromApplicationBenchmark(log_files_v1, query_files_v1)
    print(f"Real: {app_benchmark_df['mean_latency']}")
    
    print(f"Diff to real: {y_new_pred[0] - app_benchmark_df['mean_latency'][0]}")
    print(f"prediction plus mse: {y_new_pred[0] + mse}")
    print(f"prediction minus mse: {y_new_pred[0] - mse}")
    

def main():
    csv_files_v1 = getMicrobenchmarkFilePathsV104vsV105()
    csv_files_v2 = getMicrobenchmarkFilePathsV105vsV106()
    csv_files_v3 = getMicrobenchmarkFilePathsV106vsV107()

    df_v1 = load_and_clean_data(csv_files_v1)
    df_v2 = load_and_clean_data(csv_files_v2)
    df_v3 = load_and_clean_data(csv_files_v3)

    df_v1 = aggregate_data(df_v1).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v2 = aggregate_data(df_v2).pivot_table(index="version", columns='test_name', values='mean_ms_op')
    df_v3 = aggregate_data(df_v3).pivot_table(index="version", columns='test_name', values='mean_ms_op')

   # Create a DataFrame with the 'version' column
   

    # Concatenate the version DataFrame with the original DataFrame
    df_v1["version_sut"] = "v104-v105"
    df_v2["version_sut"] = "v105-v106"
    df_v3["version_sut"] = "v106-v107"

    df_v1.to_csv('loadedData_v1.csv', index=False)
    df_v2.to_csv('loadedData_v2.csv', index=False)
    df_v3.to_csv('loadedData_v3.csv', index=False)

    df_v1 = df_v1.dropna(axis=1)
    df_v2 = df_v2.dropna(axis=1)
    df_v3 = df_v3.dropna(axis=1)

    nan_indexes = df_v1[df_v1.isna().any(axis=1)].index
    print("Indexes with NaN values:", nan_indexes.tolist())
    

    benchmark_df_v1 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV104vsV105(), getQueriesCSVFilePaths())
    benchmark_df_v2 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV105vsV106(), getQueriesCSVFilePaths())
    benchmark_df_v3 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV106vsV107(), getQueriesCSVFilePaths())

    benchmark_df_v1["version_sut"] = "v104-v105"
    benchmark_df_v2["version_sut"] = "v105-v106"
    benchmark_df_v3["version_sut"] = "v106-v107"
    print(df_v1)
    print(benchmark_df_v1)
    merged_df_v1 = benchmark_df_v1.merge(df_v1, on=["version", "version_sut"])
    merged_df_v2 = benchmark_df_v2.merge(df_v2, on=["version", "version_sut"])
    merged_df_v3 = benchmark_df_v3.merge(df_v3, on=["version", "version_sut"])


    print(merged_df_v1)
    merged_df_v1.to_csv('output_merged_v4_v1.csv', index=False)
    merged_df_v2.to_csv('output_merged_v4_v2.csv', index=False)
    merged_df_v3.to_csv('output_merged_v4_v3.csv', index=False)

    merged_df_v1 = pd.concat([merged_df_v1, merged_df_v2, merged_df_v3], ignore_index=True)
    nan_indexes = merged_df_v1[merged_df_v1.isna().any(axis=1)].index
    print("Indexes with NaN values:", nan_indexes.tolist())
    merged_df_v1 = merged_df_v1.fillna(0)
    merged_df_v1.to_csv('output_merged_v4.csv', index=False)
    new_final_df = merged_df_v1.drop(columns=["version", "version_sut"])

    ridge, scaler, trained_columns, mse = apply_ridge_regression_to_all(new_final_df)
    apply_ridge_to_new_data(ridge, scaler, trained_columns, mse)

    # df_v1.to_csv('loadedData_v1.csv', index=False)
    # aggregated_data_v1 = aggregate_data(df_v1).pivot_table(index=None, columns='test_name', values='mean_ms_op')

    # aggregated_data_v1.to_csv('aggregatedData_v1.csv', index=False)
    # aggregated_data_v1["version"] = "v104-v105"

    # aggregated_data_v2 = aggregate_data(df_v2).pivot_table(index=None, columns='test_name', values='mean_ms_op')
    # aggregated_data_v2["version"] = "v105-v106"
    
    # aggregated_data_v3 = aggregate_data(df_v3).pivot_table(index=None, columns='test_name', values='mean_ms_op')
    # aggregated_data_v3["version"] = "v106-v107"

    # # print(aggregated_data_v3)

    # benchmark_df_v1 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV104vsV105(), getQueriesCSVFilePaths())
    # benchmark_df_v2 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV105vsV106(), getQueriesCSVFilePaths())
    # benchmark_df_v3 = parseLogsFromApplicationBenchmark(getApplicationBenchmarkFilePathsV106vsV107(), getQueriesCSVFilePaths())

    # benchmark_df_v1["version"] = "v104-v105"
    # benchmark_df_v2["version"] = "v105-v106"
    # benchmark_df_v3["version"] = "v106-v107"

    # merged_df_v1 = pd.concat([aggregated_data_v1, aggregated_data_v2], ignore_index=True)
    # merged_df_v1 = pd.concat([merged_df_v1, aggregated_data_v3], ignore_index=True)
    
    # merged_df_v1 = merged_df_v1.fillna(0)
    # merged_df_v1.to_csv('output_merged_v4.csv', index=False)

    # mean_v1 = benchmark_df_v1["mean_latency"].mean(numeric_only=True)
    # mean_v2 = benchmark_df_v2["mean_latency"].mean(numeric_only=True)
    # mean_v3 = benchmark_df_v3["mean_latency"].mean(numeric_only=True)


    # merged_df_v1.to_csv('output_merged_v5.csv', index=False)

    # new_merged_df_v1 = merged_df_v1.merge(benchmark_df_v1, on='version')
    # new_merged_df_v2 = merged_df_v1.merge(benchmark_df_v2, on='version')
    # new_merged_df_v3 = merged_df_v1.merge(benchmark_df_v3, on='version')

    # new_merged_df_v1.to_csv('output_merged_v6_v1.csv', index=False)
    # new_merged_df_v2.to_csv('output_merged_v6_v2.csv', index=False)
    # new_merged_df_v3.to_csv('output_merged_v6_v3.csv', index=False)

    # new_final_df = pd.concat([new_merged_df_v1, new_merged_df_v2, new_merged_df_v3], ignore_index=True)

    # new_final_df.to_csv('output_merged_v6.csv', index=False)
    # new_final_df = new_final_df.drop(columns=["version"])

    # ridge, scaler, trained_columns, mse = apply_ridge_regression_to_all(new_final_df)
    # apply_ridge_to_new_data(ridge, scaler, trained_columns, mse)


    # === new attempt

    # print(merged_df_v2)
    # latencies_queries_file_paths = getQueriesCSVFilePaths()
    # latencies_inserts_file_paths = getInsertsCSVFilePaths()
    
    # app_benchmark_df = parseLogsFromApplicationBenchmark(latencies_inserts_file_paths, latencies_queries_file_paths)
    # app_benchmark_mean = app_benchmark_df["mean_latency"].mean(numeric_only=True)
    # print(app_benchmark_mean)
    # pivoted_data["mean_latency"] = app_benchmark_mean
    # print(pivoted_data)
    # app_benchmark_df.to_csv('output_appb.csv', index=False)

    # ridge, scaler, trained_columns, mse = apply_ridge_regression_to_all(pivoted_data)
    # apply_ridge_regression_to_nonSeenData(ridge, scaler, trained_columns, mse)


    # aggregated_data = df.groupby(['test_name', 'version']).agg(
    #     mean_ms_op=('ms_op', 'mean'),
    #     median_ms_op=('ms_op', 'median'),
    #     std_ms_op=('ms_op', 'std')
    # ).reset_index()

    # pivoted_data = aggregated_data.pivot(index='version', columns='test_name', values='mean_ms_op').reset_index()
    # pivoted_data.to_csv('output_piv.csv', index=False)

    # latencies_queries_file_paths = getQueriesCSVFilePaths()
    # latencies_inserts_file_paths = getInsertsCSVFilePaths()
    # app_benchmark_df = parseLogsFromApplicationBenchmark(latencies_inserts_file_paths, latencies_queries_file_paths)

    # app_benchmark_df.to_csv('output_appb.csv', index=False)

    # final_data = pd.merge(pivoted_data, app_benchmark_df, on='version')
    # final_data.to_csv('output.csv', index=False)
    
    # ridge, scaler, trained_columns, mse = apply_ridge_regression_to_all(final_data)
    # apply_ridge_regression_to_nonSeenData(ridge, scaler, trained_columns, mse)

if __name__ == "__main__":
    main()