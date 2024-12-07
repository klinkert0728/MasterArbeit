import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read both latency files
latest_df = pd.read_csv('5/latenciesInserts_latest.csv', 
                        sep=' ',
                        header=None,
                        names=['Rows', 'Execution_Time_ms'],
                        usecols=[1])  # Only use second column

other_df = pd.read_csv('5/latenciesInserts_other.csv', 
                       sep=' ',
                       header=None,
                       names=['Rows', 'Execution_Time_ms'],
                       usecols=[1])  # Only use second column

# Add sequence number for x-axis
latest_df['Query_Number'] = range(len(latest_df))
other_df['Query_Number'] = range(len(other_df))

# Create figure
plt.figure(figsize=(12, 6))

# Create line plot
plt.plot(latest_df['Query_Number'], latest_df['Execution_Time_ms'], 
         label='Latest Version', color='blue', alpha=0.6, linewidth=1)
plt.plot(other_df['Query_Number'], other_df['Execution_Time_ms'], 
         label='Other Version', color='red', alpha=0.6, linewidth=1)

# Add moving averages for smoother trend lines
window = 50  # Size of the moving average window
latest_ma = latest_df['Execution_Time_ms'].rolling(window=window).mean()
other_ma = other_df['Execution_Time_ms'].rolling(window=window).mean()

plt.plot(latest_df['Query_Number'], latest_ma, color='darkblue', 
         linewidth=2, label=f'Latest {window}-pt Moving Avg')
plt.plot(other_df['Query_Number'], other_ma, color='darkred', 
         linewidth=2, label=f'Other {window}-pt Moving Avg')

plt.title('Query Execution Times')
plt.xlabel('Query Number')
plt.ylabel('Execution Time (ms)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Print statistics
print("\nExecution Time Statistics (ms):")
stats = pd.DataFrame({
    'Latest Version': latest_df['Execution_Time_ms'].describe(),
    'Other Version': other_df['Execution_Time_ms'].describe()
})
print(stats)

# Calculate percentage difference in mean execution time
time_diff = ((latest_df['Execution_Time_ms'].mean() - other_df['Execution_Time_ms'].mean()) 
             / other_df['Execution_Time_ms'].mean() * 100)
print(f"\nMean Execution Time Difference: {time_diff:.2f}%")
print("(Positive values indicate Latest version is slower)")

plt.tight_layout()
plt.show()

# Create a second plot with log scale
plt.figure(figsize=(12, 6))

plt.plot(latest_df['Query_Number'], latest_df['Execution_Time_ms'], 
         label='Latest Version', color='blue', alpha=0.6, linewidth=1)
plt.plot(other_df['Query_Number'], other_df['Execution_Time_ms'], 
         label='Other Version', color='red', alpha=0.6, linewidth=1)

plt.plot(latest_df['Query_Number'], latest_ma, color='darkblue', 
         linewidth=2, label=f'Latest {window}-pt Moving Avg')
plt.plot(other_df['Query_Number'], other_ma, color='darkred', 
         linewidth=2, label=f'Other {window}-pt Moving Avg')

plt.title('Query Execution Times (Log Scale)')
plt.xlabel('Query Number')
plt.ylabel('Execution Time (ms) - Log Scale')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()