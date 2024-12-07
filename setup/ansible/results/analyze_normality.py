import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
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

def analyze_normality(df, output_dir='normality_analysis'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique test names
    test_names = df['test_name'].unique()
    
    # Open a file for writing results
    with open(os.path.join(output_dir, 'normality_test_results.txt'), 'w') as f:
        f.write("Normality Test Results (Combined Data)\n")
        f.write("=" * 80 + "\n\n")
        
        for test in test_names:
            # Get data for each version
            v1_data = df[(df['test_name'] == test) & (df['version'] == 1)]['ms_op']
            v2_data = df[(df['test_name'] == test) & (df['version'] == 2)]['ms_op']
            
            if len(v1_data) < 3 or len(v2_data) < 3:
                continue  # Skip if not enough data points
            
            # Perform Shapiro-Wilk test
            _, v1_p = stats.shapiro(v1_data)
            _, v2_p = stats.shapiro(v2_data)
            
            # Write results to file
            f.write(f"Test: {test}\n")
            f.write(f"Version 1 (v1.04)\n")
            f.write(f"  Sample size: {len(v1_data)}\n")
            f.write(f"  Mean: {v1_data.mean():.4f} ms\n")
            f.write(f"  Std Dev: {v1_data.std():.4f} ms\n")
            f.write(f"  Shapiro-Wilk p-value: {v1_p:.4f} {'(Normal)' if v1_p > 0.05 else '(Not Normal)'}\n\n")
            
            f.write(f"Version 2 (Latest)\n")
            f.write(f"  Sample size: {len(v2_data)}\n")
            f.write(f"  Mean: {v2_data.mean():.4f} ms\n")
            f.write(f"  Std Dev: {v2_data.std():.4f} ms\n")
            f.write(f"  Shapiro-Wilk p-value: {v2_p:.4f} {'(Normal)' if v2_p > 0.05 else '(Not Normal)'}\n\n")
            f.write("-" * 80 + "\n\n")
            
            # Create Q-Q plots
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            stats.probplot(v1_data, dist="norm", plot=plt)
            plt.title(f"Q-Q Plot: Version 1 (v1.04)\n{test}")
            
            plt.subplot(1, 2, 2)
            stats.probplot(v2_data, dist="norm", plot=plt)
            plt.title(f"Q-Q Plot: Version 2 (Latest)\n{test}")
            
            plt.tight_layout()
            safe_test_name = test.replace('/', '_').replace('\\', '_')
            plt.savefig(os.path.join(output_dir, f'qq_plot_{safe_test_name}.png'))
            plt.close()
            
            # Create distribution plots
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=v1_data, label='Version 1 (v1.04)')
            sns.kdeplot(data=v2_data, label='Version 2 (Latest)')
            plt.title(f'Distribution Plot: {test}')
            plt.xlabel('ms/op')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'dist_plot_{safe_test_name}.png'))
            plt.close()

def main():
    # File paths
    file_paths = ['2/microbenchmark.csv', '3/microbenchmark.csv', '4/microbenchmark.csv']
    
    # Parse and combine CSVs
    df = parse_csv(file_paths)
    
    # Analyze normality
    analyze_normality(df)
    
    print("Analysis complete! Check the 'normality_analysis' directory for results.")

if __name__ == "__main__":
    main() 