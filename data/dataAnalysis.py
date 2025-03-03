import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import re
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from queriesPaths import getFilePaths, getPredictionFilePaths, getQueriesCSVFilePaths, getInsertsCSVFilePaths, getMicrobenchmarkFilePathsV104vsV105, getMicrobenchmarkFilePathsV105vsV106, getMicrobenchmarkFilePathsV106vsV107, getApplicationBenchmarkFilePathsV104vsV105, getApplicationBenchmarkFilePathsV105vsV106, getApplicationBenchmarkFilePathsV106vsV107, getApplicationBenchmarkFilePathsV107vsV108, getMicrobenchmarkFilePathsV107vsV108, getApplicationBenchmarkFilePathsV108vsV109, getMicrobenchmarkFilePathsV108vsV109
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures

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
        summary['run'] = log_file_path.split("/")[-1].split("-")[0]
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
        run = df["run"]
        file_name = file_path.split("/")[-1]
        file_name = file_name.split("-")[0]
        df["run"] = file_name
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def remove_highly_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop)


def apply_ridge_regression_to_all(cleaned_df):
    # Ensure the DataFrame is not empty
    if cleaned_df.empty:
        print("The DataFrame is empty.")
        return

    # Define features (X) and target (y)
    # Assuming 'mean_latency' is the target variable and all other numeric columns are features
    X = cleaned_df.drop(columns=["mean_latency"])
    print(X.shape)
    # X = remove_highly_correlated_features(X)
    # X = remove_perfectly_collinear_features(X)
    print(X.shape)
    y = cleaned_df['mean_latency']

    # Add feature transformation
    X_log = np.log1p(X.clip(lower=0))  # Log transform features
    y_log = np.log1p(y.clip(lower=0))  # Log transform target
    
    # Split the transformed data
    X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different alpha values
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    best_alpha = None
    best_score = float('-inf')
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5)
        mean_score = scores.mean()
        print(f"Alpha: {alpha}, Mean CV Score: {mean_score}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}")
    
    # Train final model with best alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_scaled, y_train)
    
    # Transform predictions back to original scale
    y_pred = np.expm1(ridge.predict(X_test_scaled))
    y_test_original = np.expm1(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    
    print(f"\nFinal Model Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return ridge, scaler, X_train.columns, mse


def remove_perfectly_collinear_features(dataframe):
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]

    # Identify features with high VIF
    to_drop = vif_data[vif_data["VIF"] > 10]["feature"].tolist()

    print(f"Features to drop due to high VIF: {to_drop}")
    # Drop features with high VIF
    reduced_df = dataframe.drop(columns=to_drop)
    print(f"Reduced dataframe shape: {reduced_df.shape}")
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

def transform_data(df):
    new_df = df

    # Pivot the DataFrame
    pivoted_df = new_df.pivot_table(index=['run', 'version'], columns='test_name', values='ms_op', aggfunc='first')

    # Reset the index if you want a clean DataFrame
    pivoted_df.reset_index(inplace=True)

    # Save the transformed DataFrame to a new CSV file
    pivoted_df.to_csv('transformed_file.csv', index=False)
    return pivoted_df

def apply_ridge_to_new_data(ridge, scaler, trained_columns, mse):
    csv_files_v1, log_files_v1, query_files_v1 = getPredictionFilePaths()

    df_v1 = load_and_clean_data(csv_files_v1)
    df_v1 = transform_data(df_v1)
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
    

def analyze_data_distribution(df):
    # Remove existing code...
    
    for feature in df.columns:
        if feature != 'mean_latency':
            # Basic statistics
            print(f"\nAnalysis for {feature}:")
            print(df[feature].describe())
            
            # Check for outliers using IQR method
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))]
            print(f"Number of outliers: {len(outliers)}")
            
            # Plot distribution
            plt.figure(figsize=(12, 4))
            
            # Subplot 1: Distribution plot
            plt.subplot(1, 2, 1)
            sns.histplot(df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            
            # Subplot 2: Box plot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[feature])
            plt.title(f'Box Plot of {feature}')
            
            plt.tight_layout()
            plt.savefig(f'test_data/distribution_{feature.replace("/", "_")}.png')
            plt.close()

def preprocess_features(X, y):
    """Preprocess features with more sophisticated transformations"""
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    X_filtered = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
    
    # Create polynomial features for top important features
    top_features = [
        'BenchmarkIntersectFullOverlap/items_1000-2',
        'BenchmarkMapHasHit/items_1000-2'
    ]
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    top_features_df = X_filtered[top_features]
    poly_features = pd.DataFrame(
        poly.fit_transform(top_features_df),
        columns=[f"poly_{i}" for i in range(poly.n_output_features_)]
    )
    
    # Combine original and polynomial features
    X_enhanced = pd.concat([X_filtered, poly_features], axis=1)
    
    # Log transform the target variable
    y_log = np.log1p(y)
    
    return X_enhanced, y_log

def try_different_models(X, y):
    # Preprocess features
    X_enhanced, y_log = preprocess_features(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y_log, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with hyperparameter grids
    models = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    results = {}
    feature_importance_dict = {}
    
    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        
        # Perform GridSearch with cross-validation
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions and transform back to original scale
        y_pred = np.expm1(best_model.predict(X_test_scaled))
        y_test_original = np.expm1(y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        
        results[name] = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'Best Params': grid_search.best_params_
        }
        
        print(f"\nResults for {name}:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"MSE: {mse:.2f}")
        print(f"R2: {r2:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_enhanced.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_dict[name] = feature_importance
            
            print(f"\nTop 10 Most Important Features for {name}:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
            plt.title(f'Top 20 Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(f'test_data/feature_importance_{name}.png')
            plt.close()
    
    return results, feature_importance_dict

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

    df_v1.to_csv('loadedData_withoutAggregated.csv', index=False)

    df_v1 = transform_data(df_v1)
    df_v2 = transform_data(df_v2)
    df_v3 = transform_data(df_v3)
    df_v4 = transform_data(df_v4)
    df_v5 = transform_data(df_v5)
    # Create a DataFrame with the 'version' column
   
    # Concatenate the version DataFrame with the original DataFrame
    df_v1["version_sut"] = "v104-v105"
    df_v2["version_sut"] = "v105-v106"
    df_v3["version_sut"] = "v106-v107"
    df_v4["version_sut"] = "v107-v108"
    df_v5["version_sut"] = "v108-v109"

    df_v1.to_csv('loadedData_v1.csv', index=False)
    df_v2.to_csv('loadedData_v2.csv', index=False)
    df_v3.to_csv('loadedData_v3.csv', index=False)
    df_v4.to_csv('loadedData_v4.csv', index=False)
    df_v5.to_csv('loadedData_v5.csv', index=False)

    df_v1 = df_v1.dropna(axis=1)
    df_v2 = df_v2.dropna(axis=1)
    df_v3 = df_v3.dropna(axis=1)
    df_v4 = df_v4.dropna(axis=1)
    df_v5 = df_v5.dropna(axis=1)

    nan_indexes = df_v1[df_v1.isna().any(axis=1)].index
    print("Indexes with NaN values:", nan_indexes.tolist())
    
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

    print(df_v1)
    print(benchmark_df_v1)
    merged_df_v1 = benchmark_df_v1.merge(df_v1, on=["version", "version_sut", 'run'])
    merged_df_v2 = benchmark_df_v2.merge(df_v2, on=["version", "version_sut", 'run'])
    merged_df_v3 = benchmark_df_v3.merge(df_v3, on=["version", "version_sut", 'run'])
    merged_df_v4 = benchmark_df_v4.merge(df_v4, on=["version", "version_sut", 'run'])
    merged_df_v5 = benchmark_df_v5.merge(df_v5, on=["version", "version_sut", 'run'])

    print(merged_df_v1)
    merged_df_v1.to_csv('output_merged_v4_v1.csv', index=False)
    merged_df_v2.to_csv('output_merged_v4_v2.csv', index=False)
    merged_df_v3.to_csv('output_merged_v4_v3.csv', index=False)
    merged_df_v4.to_csv('output_merged_v4_v4.csv', index=False)
    merged_df_v5.to_csv('output_merged_v4_v5.csv', index=False)

    merged_df_v1 = pd.concat([merged_df_v1, merged_df_v2, merged_df_v3, merged_df_v4, merged_df_v5], ignore_index=True)
    nan_indexes = merged_df_v1[merged_df_v1.isna().any(axis=1)].index
    merged_df_v1 = merged_df_v1.fillna(0)
    merged_df_v1.to_csv('output_merged_v4.csv', index=False)
    new_final_df = merged_df_v1.drop(columns=["version", "version_sut", "run"])

    plot_features_against_mean_latency(new_final_df)
    ridge, scaler, trained_columns, mse = apply_ridge_regression_to_all(new_final_df)
    apply_ridge_to_new_data(ridge, scaler, trained_columns, mse)

    # Add these lines after creating new_final_df
    print("\nAnalyzing data distribution...")
    analyze_data_distribution(new_final_df)
    
    print("\nTrying different models...")
    X = new_final_df.drop(columns=['mean_latency'])
    y = new_final_df['mean_latency']
    model_results, feature_importance_dict = try_different_models(X, y)

def plot_features_against_mean_latency(df):
    # Ensure 'mean_latency' is in the DataFrame
    if 'mean_latency' not in df.columns:
        print("The DataFrame does not contain 'mean_latency'.")
        return

    # Iterate over each feature in the DataFrame
    for feature in df.columns:
        if feature != 'mean_latency':
            plt.figure(figsize=(8, 6))
            
            plt.scatter(df[feature], df['mean_latency'], alpha=0.5)
            plt.title(f'{feature} vs Mean Latency')
            plt.xlabel(feature)
            plt.ylabel('Mean Latency')
            plt.grid(True)
            plt.savefig(f'test_data/{feature.replace("/", "_")}_vs_mean_latency.png')
            plt.close()

if __name__ == "__main__":
    main()