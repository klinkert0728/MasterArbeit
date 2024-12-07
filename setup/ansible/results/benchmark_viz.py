import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file directly with pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv('1/microbenchmark.csv', 
                 sep=';',
                 names=['run', 'type', 'path', 'benchmark', 'version', 'iterations', 'runtime'])

# Clean up the benchmark names by removing the -2 suffix
#

# Calculate average runtime for each benchmark and version
grouped_avg = df.groupby(['benchmark', 'version'])['runtime'].mean().unstack()

# Create scatter plot
plt.figure(figsize=(15, 10))
plt.scatter(grouped_avg[1], grouped_avg[2], alpha=0.6)

# Add diagonal line for reference (y=x)
max_val = max(grouped_avg[1].max(), grouped_avg[2].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Performance')

plt.xlabel('Latest Version Runtime (ns)')
plt.ylabel('Other Version Runtime (ns)')
plt.title('Benchmark Runtime Comparison\n(Average of 3 runs per version)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log')
plt.yscale('log')

# Calculate percentage differences
percent_diff = (grouped_avg[2] - grouped_avg[1]) / grouped_avg[1] * 100
significant_diff = percent_diff[abs(percent_diff) > 5]
significant_diff_sorted = significant_diff.sort_values(key=abs, ascending=False)

# Add annotations for significant differences
for idx in significant_diff_sorted.index:
    plt.annotate(f'{idx.split("/")[-1]}\n({percent_diff[idx]:.1f}%)', 
                (grouped_avg[1][idx], grouped_avg[2][idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, alpha=0.7)

plt.legend()

# Print detailed analysis
print("\nPerformance Summary:")
print(f"Total unique benchmarks: {len(grouped_avg)}")
print(f"Benchmarks with >5% difference: {len(significant_diff_sorted)}")

print("\nDetailed differences (>5% change):")
print("\nBenchmark | Latest (ns) | Other (ns) | Difference (%)")
print("-" * 80)
for idx in significant_diff_sorted.index:
    latest = grouped_avg[1][idx]
    other = grouped_avg[2][idx]
    diff = significant_diff_sorted[idx]
    print(f"{idx.split('/')[-1][:40]:<40} | {latest:>10.2f} | {other:>10.2f} | {diff:>10.1f}%")

# Verify we have exactly 3 runs per benchmark per version
run_counts = df.groupby(['benchmark', 'version']).size()
if not all(run_counts == 3):
    print("\nWarning: Not all benchmarks have exactly 3 runs!")
    print(run_counts[run_counts != 3])

# Example verification
print("\nVerification for BenchmarkRowsUnmarshal-2:")
benchmark_data = df[df['benchmark'].str.contains('BenchmarkRowsUnmarshal-2')]
print(benchmark_data)
print("\nVersion 1 (Latest) runs:", benchmark_data[benchmark_data['version'] == '1']['runtime'].tolist())
print("Version 1 average:", benchmark_data[benchmark_data['version'] == '1']['runtime'].mean())
print("\nVersion 2 (Other) runs:", benchmark_data[benchmark_data['version'] == '2']['runtime'].tolist())
print("Version 2 average:", benchmark_data[benchmark_data['version'] == '2']['runtime'].mean())

plt.tight_layout()
plt.show() 