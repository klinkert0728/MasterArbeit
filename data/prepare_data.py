import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.linear_model import Ridge

def parse_csv(file_paths):
    # Read and combine multiple CSV files
    columns = ['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op']
    dfs = [pd.read_csv(file_path, sep=';', names=columns) for file_path in file_paths]
    return pd.concat(dfs, ignore_index=True)

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

def train(X, Y):
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Train the model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    # Make predictions
    Y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

def prepare_model(final_data):
    # Define X (microbenchmark features) and Y (target variable, e.g., 'mean_latency')
    X = final_data.drop(['version', 'mean_latency'], axis=1)
    Y = final_data['mean_latency']  # Target variable can be switched as needed
    # Inspect X and Y to ensure proper setup
    # print("Features (X):")
    # print(X.head())
    # print("Target (Y):")
    # print(Y.head())
    return X, Y

def apply_pca(X, n_components=3):
    # Apply PCA to reduce the dimensionality of X to n_components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance by the first {n_components} components: {pca.explained_variance_ratio_}")
    return X_pca

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

def calculate_vif(X):
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def train_and_predict_with_ridge(X, Y):
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train the Ridge regression model
    ridge_model = Ridge(alpha=0.5)  # You can adjust the alpha parameter
    ridge_model.fit(X_train, Y_train)
    
    # Make predictions on the test set
    Y_pred = ridge_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')
    
    return Y_pred

def main():
    # File paths for microbenchmark CSVs
    csv_file_paths = ['2-microbenchmark.csv', '3-microbenchmark.csv', '4-microbenchmark.csv']  # Add more CSV files as needed
    log_file_paths = ['2-logInserts_other.log', '3-logInserts_other.log', '4-logInserts_other.log', '2-logInserts_latest.log', '3-logInserts_latest.log', '4-logInserts_latest.log']
    
    # Parse and combine CSVs
    df = parse_csv(csv_file_paths)
    
    # Analyze normality
    aggregated_data = df.groupby(['test_name', 'version']).agg(
        mean_ms_op=('ms_op', 'mean'),
        median_ms_op=('ms_op', 'median'),
        std_ms_op=('ms_op', 'std')
    ).reset_index()

    # Pivot the table so that each test_name becomes a column with its mean value
    pivoted_data = aggregated_data.pivot(index='version', columns='test_name', values='mean_ms_op').reset_index()

    # Extract application benchmark results from each log file
    app_benchmark_df = parseLogsFromApplicationBenchmark(log_file_paths)

    # Merge the pivoted microbenchmark data with the application benchmark data
    final_data = pd.merge(pivoted_data, app_benchmark_df, on='version')

    # Prepare the model data
    X, Y = prepare_model(final_data)

    # Apply PCA to reduce the dimensionality of X
    X_pca = apply_pca(X, n_components=3)

    # Check for constant features
    constant_features = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_features)

    # Check for perfect collinearity
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

    # Find features with perfect correlation
    perfect_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] == 1)]

    # Drop one of each pair of perfectly correlated features
    X = X.drop(columns=perfect_corr_features)

    #Calculate VIF
    vif_data = calculate_vif(X)

    # Remove features with high VIF
    while vif_data['VIF'].max() > 5:
        feature_to_remove = vif_data.sort_values('VIF', ascending=False).iloc[0]['feature']
        X = X.drop(columns=[feature_to_remove])
        vif_data = calculate_vif(X)

    # # Train the model using the principal components
    # print(X.head())
    train_ridge_cv(X, Y)
    train(X, Y)

if __name__ == "__main__":
    main() 