import pandas as pd
import matplotlib.pyplot as plt

def analyze_benchmarks(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, sep=';', header=None,
                     names=['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op'])
    
    # Group by test_name and version, calculate mean ms_op
    results = df.groupby(['test_name', 'version'])['ms_op'].agg(['mean', 'std']).reset_index()
    
    # Create scatter plot
    plt.figure(figsize=(15, 10))
    
    # Plot Version 1 (v1.04) vs Version 2 (Latest)
    v1_data = results[results['version'] == 1]['mean']
    v2_data = results[results['version'] == 2]['mean']
    test_names = results[results['version'] == 1]['test_name']
    
    plt.scatter(range(len(v1_data)), v1_data, label='Version 1 (v1.04)', alpha=0.6)
    plt.scatter(range(len(v2_data)), v2_data, label='Version 2 (Latest)', alpha=0.6)
    
    plt.xticks(range(len(test_names)), test_names, rotation=45, ha='right')
    plt.ylabel('Average ms/op')
    plt.title('Benchmark Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig('benchmark_comparison.png')
    
    # Print numerical results
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    
    faster_count = {'v1.04': 0, 'Latest': 0, 'Tie': 0}
    total_diff_percent = 0
    total_benchmarks = 0
    
    for test in test_names.unique():
        total_benchmarks += 1
        v1_mean = results[(results['test_name'] == test) & (results['version'] == 1)]['mean'].values[0]
        v2_mean = results[(results['test_name'] == test) & (results['version'] == 2)]['mean'].values[0]
        diff_percent = ((v2_mean - v1_mean) / v1_mean) * 100
        
        # Determine winner for this benchmark
        if abs(diff_percent) < 0.1:  # If difference is less than 0.1%, consider it a tie
            winner = "Tie"
            faster_count['Tie'] += 1
        elif diff_percent > 0:
            winner = "v1.04"
            faster_count['v1.04'] += 1
        else:
            winner = "Latest"
            faster_count['Latest'] += 1
        
        print(f"\nTest: {test}")
        print(f"Version 1 (v1.04) average: {v1_mean:.2f} ms/op")
        print(f"Version 2 (Latest) average: {v2_mean:.2f} ms/op")
        print(f"Difference: {diff_percent:.2f}% ({'slower' if diff_percent > 0 else 'faster'})")
        print(f"Winner: {winner}")
        
        total_diff_percent += diff_percent
    
    print("\n" + "=" * 80)
    print("\nOverall Summary:")
    print(f"v1.04 was faster in {faster_count['v1.04']} benchmarks")
    print(f"Latest was faster in {faster_count['Latest']} benchmarks")
    print(f"Ties: {faster_count['Tie']} benchmarks")
    
    avg_diff = total_diff_percent / total_benchmarks
    print(f"\nOn average, Latest version was {abs(avg_diff):.2f}% {'slower' if avg_diff > 0 else 'faster'} than v1.04")
    
    # Determine overall winner
    if faster_count['v1.04'] == faster_count['Latest']:
        print("\nOverall Result: TIE - Both versions won in equal number of benchmarks")
    else:
        winner = "v1.04" if faster_count['v1.04'] > faster_count['Latest'] else "Latest"
        print(f"\nOverall Winner: {winner} version")
        print(f"Won {max(faster_count['v1.04'], faster_count['Latest'])} out of {total_benchmarks} benchmarks")
        print(f"(Ties: {faster_count['Tie']}, Other version: {min(faster_count['v1.04'], faster_count['Latest'])})")

if __name__ == "__main__":
    analyze_benchmarks('4/microbenchmark.csv')