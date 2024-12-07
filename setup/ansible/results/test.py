# ... existing imports ...
from scipy import stats
import seaborn as sns

def analyze_benchmarks(file_path):
    # ... existing code until results calculation ...
    
    # For each unique test, perform normality test
    print("\nNormality Test Results:")
    print("=" * 80)
    
    for test in test_names.unique():
        v1_data = df[(df['test_name'] == test) & (df['version'] == 1)]['ms_op']
        v2_data = df[(df['test_name'] == test) & (df['version'] == 2)]['ms_op']
        
        # Shapiro-Wilk test for normality
        _, v1_p = stats.shapiro(v1_data)
        _, v2_p = stats.shapiro(v2_data)
        
        print(f"\nTest: {test}")
        print(f"Version 1 (v1.04) p-value: {v1_p:.4f} {'(Normal)' if v1_p > 0.05 else '(Not Normal)'}")
        print(f"Version 2 (Latest) p-value: {v2_p:.4f} {'(Normal)' if v2_p > 0.05 else '(Not Normal)'}")
        
        # Create Q-Q plots
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        stats.probplot(v1_data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: {test}\nVersion 1 (v1.04)")
        
        plt.subplot(1, 2, 2)
        stats.probplot(v2_data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: {test}\nVersion 2 (Latest)")
        
        plt.tight_layout()
        plt.savefig(f'qq_plot_{test.replace("/", "_")}.png')
        plt.close()
        
        # Create distribution plots
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=v1_data, label='Version 1 (v1.04)')
        sns.kdeplot(data=v2_data, label='Version 2 (Latest)')
        plt.title(f'Distribution Plot: {test}')
        plt.xlabel('ms/op')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'dist_plot_{test.replace("/", "_")}.png')
        plt.close()
    
    # ... rest of existing code ...