import pandas as pd
import numpy as np
import os

def parse_csv(file_paths):
    # Read and combine multiple CSV files
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=';', header=None,
                        names=['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op'])
        dfs.append(df)
    
    # Concatenate all dataframes
    return pd.concat(dfs, ignore_index=True)

def prepare_regression_data(df, output_dir='regression_analysis'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Pivot the data to have benchmarks as rows and versions as columns
    pivot_df = df.pivot_table(index='test_name', columns='version', values='ms_op', aggfunc='mean')
    
    # Rename columns for clarity
    pivot_df.columns = [f'version_{int(col)}' for col in pivot_df.columns]
    
    # Reset index to make 'test_name' a column
    pivot_df.reset_index(inplace=True)
    
    # Save to CSV
    pivot_df.to_csv(os.path.join(output_dir, 'regression_data.csv'), index=False)
    
    # Print basic information
    print("\nDataset Shape:", pivot_df.shape)
    print("\nFirst few rows:")
    print(pivot_df.head())
    print("\nBasic Statistics:")
    print(pivot_df.describe())
    
    return pivot_df

def main():
    # File paths
    file_paths = ['3/microbenchmark.csv', '4/microbenchmark.csv']
    
    # Parse and combine CSVs
    df = parse_csv(file_paths)
    
    # Prepare regression data
    regression_df = prepare_regression_data(df)
    
    print("\nAnalysis complete! Check the 'regression_analysis' directory for results.")

if __name__ == "__main__":
    main() 