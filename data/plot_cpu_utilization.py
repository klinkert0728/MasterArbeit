import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_cpu_utilization(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, skiprows=4)  # Skip the first 4 rows with metadata
    
    # Convert CPU utilization to percentage
    cpu_values = df.iloc[:, 1].values * 100  # Second column contains CPU values
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot CPU utilization
    plt.plot(range(len(cpu_values)), cpu_values, marker='o', linestyle='-', markersize=2)
    plt.xlabel('Time Points')
    plt.ylabel('CPU Utilization (%)')
    plt.title('CPU Utilization Over Time')
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('cpu_utilization_analysis.png')
    plt.close()

    # Print statistics
    print(f"Total data points: {len(cpu_values)}")
    print(f"Average CPU utilization: {np.mean(cpu_values):.2f}%")
    print(f"Maximum CPU utilization: {np.max(cpu_values):.2f}%")
    print(f"Minimum CPU utilization: {np.min(cpu_values):.2f}%")

if __name__ == "__main__":
    csv_file = "CPU_Utilization.csv"
    plot_cpu_utilization(csv_file) 