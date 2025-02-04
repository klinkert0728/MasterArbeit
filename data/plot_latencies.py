import os
import re
import matplotlib.pyplot as plt
import sys
from queriesPaths import getLatenciesFilePathsCSV
import numpy as np

def extract_mean_latency(file_paths):
    # Sample data from the snippets
    logInsertSummaryResult = []
    logInsertMapWithRunNumber = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read()

        sut_version = file_path.split('/')[-3]
        file_name = os.path.basename(file_path)
        runNumber = file_name.split('-')[0]
        run_version = 0

        if file_path.endswith('other.csv'):
            run_version = 1
        else:
            run_version = 2

        logInsertSummary = logInsertMapWithRunNumber.get(runNumber, None)
            
        if logInsertSummary is None:
            logInsertSummary = []
        
        currentLogInsertSummary = {}

        values = []

        for line in content.splitlines():
            # Split the line into columns based on whitespace
            columns = line.split()
            values.append(float(columns[1]) / 1000)

        currentLogInsertSummary["runNumber"] = runNumber
        currentLogInsertSummary["version"] = run_version
        currentLogInsertSummary["sut_version"] = sut_version
        currentLogInsertSummary["values"] = values
        logInsertSummary.append(currentLogInsertSummary)
        logInsertMapWithRunNumber[runNumber] = logInsertSummary
       
    return logInsertMapWithRunNumber


def plot_mean_latencies(log_file_paths):
    version_means = extract_mean_latency(log_file_paths)
    available_sut_versions = set()
    for runNumber, means in version_means.items():
        means = sorted(means, key=lambda x: x['sut_version'])
        available_sut_versions.add(means[0]['sut_version'])
 
        # Plot mean latency for each run with bars for both versions
    # Prepare data for plotting
    print(available_sut_versions)
    sut_means = {}
    for sut_version in available_sut_versions:
        # Prepare data for plotting
        for runNumber, means in version_means.items():
            means = [entry for entry in means if entry['sut_version'] == sut_version]
            logInsertSummary = sut_means.get(sut_version, None)
            if logInsertSummary is None:
                logInsertSummary = []

            
            if len(means) > 0:
                logInsertSummary.append(means)
                sut_means[sut_version] = logInsertSummary

    for sut_version, means in sut_means.items():
        
        run_numbers = list(range(len(means)))
        print(run_numbers)
        version_1_means = [next((entry['values'] for entry in means if entry['version'] == 1), 0) for means in means]
        version_2_means = [next((entry['values'] for entry in means if entry['version'] == 2), 0) for means in means]

        for test1 in version_1_means:
            print(test1.items().keys())
            break

        break
        
        # Create a grouped bar plot
        x = np.arange(len(run_numbers))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        bars1 = ax.bar(x - width/2, version_1_means, width, label=f'{sut_version.split("-")[0]}')
        bars2 = ax.bar(x + width/2, version_2_means, width, label=f'{sut_version.split("-")[1]}')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title(f'Mean Latency by Version and Run for {sut_version}')
        ax.set_xticks(x)
        ax.set_xticklabels(run_numbers)
        ax.legend()

        # Add a grid
        ax.grid(True)

        # Save the plot
        plt.savefig(f'mean_latency_all_runs-{sut_version}.png')
        plt.close()


if __name__ == "__main__":
    # Example log_file_paths, replace with actual paths from queriesPaths.py
    stats_file_paths = getLatenciesFilePathsCSV()

    result = plot_mean_latencies(stats_file_paths)
    # plot_mean_latencies(log_file_paths)
