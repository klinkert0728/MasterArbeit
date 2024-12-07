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


def parseLogsFromApplicationBenchmark(log_file_paths):
    app_benchmark_summaries = []
    for log_file_path in log_file_paths:
        summary = extract_summary_metrics(log_file_path)
        if log_file_path.endswith('other.log'):
            summary['version'] = 1
        else:
            summary['version'] = 2

        # summary['run'] = log_file_path.split('-')[0]
        app_benchmark_summaries.append(summary)
    
    return pd.DataFrame(app_benchmark_summaries)

def load_and_clean_data(csv_file_paths):
    # Load and combine data from multiple CSV files
    columns = ['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op']
    dfs = []
    for file_path in csv_file_paths:
        newPat = "../setup/"
        df = pd.read_csv(file_path, sep=';', names=columns)
        # Replace triple quotes with an empty string in the entire DataFrame
        df = df.applymap(lambda x: x.replace('"""', '').replace('"', '') if isinstance(x, str) else x)
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

def apply_linear_regression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)
    Y_pred = linear_model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    return linear_model


def explore_data(X):
    print("Data Head:")
    print(X.head())
    print("\nData Description:")
    print(X.describe())
    print("\nCorrelation Matrix:")
    print(X.corr())

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

    # Train the Ridge Regression model
    ridge = Ridge(alpha=1.0)  # You can adjust the alpha parameter
    ridge.fit(X_train_scaled, y_train)

    # Predict and evaluate the model
    y_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Ridge Regression Coefficients: {ridge.coef_}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


def remove_perfectly_collinear_features(dataframe):
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr().abs()

    # Identify features with perfect collinearity
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] == 1)]

    # Drop one of each pair of perfectly collinear features
    reduced_df = dataframe.drop(columns=to_drop)
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

    # for i in range(num_groups):
    #     # Determine the start and end indices for the current group
    #     start_idx = i * group_size
    #     end_idx = min((i + 1) * group_size, num_features)

    #     # Select the subset of features for the current group
    #     group_features = features[start_idx:end_idx]

    #     # Extract the sub-correlation matrix for the current group
    #     sub_corr_matrix = corr_matrix.loc[group_features, group_features]

    #     # Plot the heatmap for the current group
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(sub_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    #     plt.title(f'Correlation Matrix for Features {start_idx + 1} to {end_idx}')
    #     plt.show()

def main():
    csv_file_paths = ['../setup/ansible/results/1/1-microbenchmark.csv', '../setup/ansible/results/2/2-microbenchmark.csv', '../setup/ansible/results/3/3-microbenchmark.csv', '../setup/ansible/results/4/4-microbenchmark.csv', '../setup/ansible/results/5/5-microbenchmark.csv', '../setup/ansible/results/6/6-microbenchmark.csv', '../setup/ansible/results/7/7-microbenchmark.csv']
    log_file_paths = ['../setup/ansible/results/1/1-logInserts_other.log', '../setup/ansible/results/1/1-logInserts_latest.log', '../setup/ansible/results/2/2-logInserts_other.log', '../setup/ansible/results/2/2-logInserts_latest.log', '../setup/ansible/results/3/3-logInserts_other.log', '../setup/ansible/results/3/3-logInserts_latest.log', '../setup/ansible/results/4/4-logInserts_other.log', '../setup/ansible/results/4/4-logInserts_latest.log', '../setup/ansible/results/5/5-logInserts_other.log', '../setup/ansible/results/5/5-logInserts_latest.log', '../setup/ansible/results/6/6-logInserts_other.log', '../setup/ansible/results/6/6-logInserts_latest.log', '../setup/ansible/results/7/7-logInserts_other.log', '../setup/ansible/results/7/7-logInserts_latest.log']

    df = load_and_clean_data(csv_file_paths)
    aggregated_data = df.groupby(['test_name', 'version']).agg(
        mean_ms_op=('ms_op', 'mean'),
        median_ms_op=('ms_op', 'median'),
        std_ms_op=('ms_op', 'std')
    ).reset_index()

    pivoted_data = aggregated_data.pivot(index='version', columns='test_name', values='mean_ms_op').reset_index()
    pivoted_data.to_csv('output_piv.csv', index=False)
    app_benchmark_df = parseLogsFromApplicationBenchmark(log_file_paths)
    app_benchmark_df.to_csv('output_appb.csv', index=False)
    final_data = pd.merge(pivoted_data, app_benchmark_df, on='version')

    X = final_data.drop(['version', 'mean_latency', 'min_latency', 'med_latency', 'max_latency', 'stddev_latency', 'total_latency_time'], axis=1)
    Y = final_data['mean_latency']

    # X_reduced = remove_highly_correlated_features(X)
    final_data.to_csv('output.csv', index=False)

    X.to_csv('X.csv', index=False)
    apply_ridge_regression_to_all(final_data)
    
    plot_grouped_heatmaps(final_data)

    # # Visualize data
    #visualize_data(X, Y)

    # # Standardize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # #explore_data(final_data)

    # # # Apply Linear Regression
    # apply_linear_regression(X_scaled, Y)


if __name__ == "__main__":
    main()