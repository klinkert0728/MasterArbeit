import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_clean_data(csv_file_paths):
    # Load and combine data from multiple CSV files
    columns = ['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op']
    dfs = []
    for file_path in csv_file_paths:
        df = pd.read_csv(file_path, sep=';', names=columns)
        # Replace triple quotes with an empty string in the entire DataFrame
        df = df.applymap(lambda x: x.replace('"""', '').replace('"', '') if isinstance(x, str) else x)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
    # grouped = combined_df.groupby(['test_name', 'version'])

    # return grouped

def plot_points_for_test_name(cleaned_df, test_name):
    # Filter the DataFrame to include only version 1 and the specified test_name
    version_1_df = cleaned_df[(cleaned_df['version'] == 1) & (cleaned_df['test_name'] == test_name)]

    if version_1_df.empty:
        print(f"No data found for test_name: {test_name} with version 1.")
        return

    # Plot the points for the 'ms_op' column
    plt.figure(figsize=(8, 6))
    plt.scatter(version_1_df.index, version_1_df['ms_op'], color='blue', alpha=0.6)
    plt.title(f'Data Points for {test_name}, version: 1')
    plt.xlabel('Index')
    plt.ylabel('ms_op')
    plt.grid(True)
    plt.show()

def evaluate_collinearity_by_test_name(cleaned_df):
    # Define the directory for saving plots
    plots_dir = 'plots'
    
    # Create the plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Group by test_name and version
    #grouped = cleaned_df.groupby(['test_name', 'version'])
    version_1_df = cleaned_df[cleaned_df['version'] == 1]
    grouped = version_1_df.groupby('test_name')

    for (test_name, version), group in grouped:
        print(f"Evaluating collinearity for test: {test_name}, version: {version}")

        # Select only numeric columns for correlation
        numeric_cols = group.select_dtypes(include=[float, int]).columns
        if 'ms_op' in numeric_cols:
            numeric_cols = ['ms_op']  # Focus on ms_op for correlation

        # Calculate the correlation matrix
        corr_matrix = group[numeric_cols].corr()

        # Plot the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix for {test_name}, version: {version}')
        plt.savefig(os.path.join(plots_dir, f'corr_matrix_{test_name.replace("/", "_")}_{version}.png'))
        plt.close()

        # Plot pair plots for numeric columns
        pairplot = sns.pairplot(group[numeric_cols])
        pairplot.fig.suptitle(f'Pair Plot of Features for {test_name}, version: {version}', y=1.02)
        pairplot.savefig(os.path.join(plots_dir, f'pair_plot_{test_name.replace("/", "_")}_{version}.png'))
        plt.close()

def calculate_vif(dataframe):
    # Select only numeric columns for VIF calculation
    numeric_cols = dataframe.select_dtypes(include=[float, int]).columns
    X = dataframe[numeric_cols]

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data

def evaluate_vif_by_test_name(cleaned_df):
    # Group by test_name and version
    grouped = cleaned_df.groupby(['test_name', 'version'])

    for (test_name, version), group in grouped:
        print(f"Evaluating VIF for test: {test_name}, version: {version}")

        # Calculate VIF
        vif_data = calculate_vif(group)

        # Print VIF results
        print(vif_data)

def main():
    # Load and combine data
    csv_file_paths = ['1-microbenchmark.csv', '2-microbenchmark.csv', '3-microbenchmark.csv', '4-microbenchmark.csv', '5-microbenchmark.csv', '6-microbenchmark.csv', '7-microbenchmark.csv']
    cleaned_data = load_and_clean_data(csv_file_paths)
    print(cleaned_data)
    # Evaluate collinearity by test_name
    evaluate_collinearity_by_test_name(cleaned_data)
    # Evaluate VIF by test_name
    evaluate_vif_by_test_name(cleaned_data)

    # Plot points for test_name
    plot_points_for_test_name(cleaned_data, 'common_string_values-2')

if __name__ == "__main__":
    main()