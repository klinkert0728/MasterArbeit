import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV files
latest_df = pd.read_csv('5/logInserts_latest.csv')
other_df = pd.read_csv('5/logInserts_other.csv')

# Convert time to relative seconds from start
start_time = min(latest_df['time'].min(), other_df['time'].min())
latest_df['relative_time'] = latest_df['time'] - start_time
other_df['relative_time'] = other_df['time'] - start_time

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Insert Performance Comparison', fontsize=14)

# Plot 1: Metrics per second
ax1.scatter(latest_df['relative_time'], 
           latest_df['instant_metrics_per_sec'], 
           alpha=0.5, label='Latest Version', color='blue', s=30)
ax1.scatter(other_df['relative_time'], 
           other_df['instant_metrics_per_sec'], 
           alpha=0.5, label='Other Version', color='red', s=30)

ax1.set_title('Instant Metrics Insert Rate')
ax1.set_xlabel('Time (seconds from start)')
ax1.set_ylabel('Metrics per Second')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Rows per second
ax2.scatter(latest_df['relative_time'], 
           latest_df['instant_rows_per_sec'], 
           alpha=0.5, label='Latest Version', color='blue', s=30)
ax2.scatter(other_df['relative_time'], 
           other_df['instant_rows_per_sec'], 
           alpha=0.5, label='Other Version', color='red', s=30)

ax2.set_title('Instant Row Insert Rate')
ax2.set_xlabel('Time (seconds from start)')
ax2.set_ylabel('Rows per Second')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print statistics
print("\nPerformance Statistics:")
stats = {
    'Metric Rate (Latest)': latest_df['instant_metrics_per_sec'].describe(),
    'Metric Rate (Other)': other_df['instant_metrics_per_sec'].describe(),
    'Row Rate (Latest)': latest_df['instant_rows_per_sec'].describe(),
    'Row Rate (Other)': other_df['instant_rows_per_sec'].describe()
}
stats_df = pd.DataFrame(stats)
print(stats_df)

# Calculate percentage differences
avg_metric_rate_diff = ((latest_df['instant_metrics_per_sec'].mean() - other_df['instant_metrics_per_sec'].mean()) 
                       / other_df['instant_metrics_per_sec'].mean() * 100)
avg_row_rate_diff = ((latest_df['instant_rows_per_sec'].mean() - other_df['instant_rows_per_sec'].mean()) 
                    / other_df['instant_rows_per_sec'].mean() * 100)

print("\nPerformance Differences:")
print(f"Average Metric Rate Difference: {avg_metric_rate_diff:.2f}%")
print(f"Average Row Rate Difference: {avg_row_rate_diff:.2f}%")
print("(Negative values indicate Latest version is slower)")

# Create statistical comparison
print("\nPerformance Statistics:")
stats = {
    'Metric Rate (Latest)': latest_df['instant_metrics_per_sec'].describe(),
    'Metric Rate (Other)': other_df['instant_metrics_per_sec'].describe(),
    'Row Rate (Latest)': latest_df['instant_rows_per_sec'].describe(),
    'Row Rate (Other)': other_df['instant_rows_per_sec'].describe()
}
stats_df = pd.DataFrame(stats)
print(stats_df)

# Calculate percentage differences
avg_metric_rate_diff = ((latest_df['instant_metrics_per_sec'].mean() - other_df['instant_metrics_per_sec'].mean()) 
                       / other_df['instant_metrics_per_sec'].mean() * 100)
avg_row_rate_diff = ((latest_df['instant_rows_per_sec'].mean() - other_df['instant_rows_per_sec'].mean()) 
                    / other_df['instant_rows_per_sec'].mean() * 100)

print("\nPerformance Differences:")
print(f"Average Metric Rate Difference: {avg_metric_rate_diff:.2f}%")
print(f"Average Row Rate Difference: {avg_row_rate_diff:.2f}%")
print("(Negative values indicate Latest version is slower)")

# Optional: Create box plots for rate distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([latest_df['instant_metrics_per_sec'], other_df['instant_metrics_per_sec']], 
           labels=['Latest', 'Other'])
plt.title('Metric Rate Distribution')
plt.ylabel('Metrics per Second')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.boxplot([latest_df['instant_rows_per_sec'], other_df['instant_rows_per_sec']], 
           labels=['Latest', 'Other'])
plt.title('Row Rate Distribution')
plt.ylabel('Rows per Second')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Extract benchmark names for microbenchmark only
micro_df['BenchmarkName'] = micro_df['Benchmark'].str.extract(r'/([^/]+)$')

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Performance Comparison: Latest vs Other', fontsize=16)

# Plot Microbenchmarks
colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
micro_benchmarks = micro_df['BenchmarkName'].unique()
for bench, color in zip(micro_benchmarks, colors):
    bench_data = micro_df[micro_df['BenchmarkName'] == bench]
    ax1.scatter(bench_data['Invocations'], bench_data['Runtime'], 
               alpha=0.6, 
               label=bench,
               color=color)

ax1.set_xlabel('Invocations')
ax1.set_ylabel('Runtime (ns)')
ax1.set_title('Microbenchmark Latencies')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot Metrics Rate Comparison
ax2.plot(latest_df['relative_time'].astype(np.float64), 
         latest_df['per_metric_rate'].astype(np.float64), 
         label='Latest Version', color='blue', alpha=0.6)
ax2.plot(other_df['relative_time'].astype(np.float64), 
         other_df['per_metric_rate'].astype(np.float64), 
         label='Other Version', color='red', alpha=0.6)
ax2.set_xlabel('Time (seconds from start)')
ax2.set_ylabel('Metrics per Second')
ax2.set_title('Metric Insert Rate')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Plot Row Rate Comparison
ax3.plot(latest_df['relative_time'].astype(np.float64), 
         latest_df['per_row_rate'].astype(np.float64), 
         label='Latest Version', color='blue', alpha=0.6)
ax3.plot(other_df['relative_time'].astype(np.float64), 
         other_df['per_row_rate'].astype(np.float64), 
         label='Other Version', color='red', alpha=0.6)
ax3.set_xlabel('Time (seconds from start)')
ax3.set_ylabel('Rows per Second')
ax3.set_title('Row Insert Rate')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Create box plot comparison
plt.figure(figsize=(10, 6))
plt.boxplot([latest_df['Latency'], other_df['Latency']], 
           labels=['Latest Version', 'Other Version'])
plt.title('Insert Operation Latency Distribution')
plt.ylabel('Latency (ms)')
plt.yscale('log')  # Use log scale for latency
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Calculate an