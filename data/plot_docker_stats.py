import os
import re
import matplotlib.pyplot as plt
import sys
from queriesPaths import getStatsFilePaths
import numpy as np

def extract_mean_latency(file_paths):
    # Sample data from the snippets
    stats_data = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read()

        sut_version = file_path.split('/')[-2]
        # Iterate over each line in the data
        summary = stats_data.get(sut_version, None)
        if summary is None:
            summary = {}

        for line in content.splitlines():
            # Split the line into columns based on whitespace
            columns = line.split()
            # Check if the line has at least 3 columns
            if len(columns) >= 3:
                # Print the first and third columns
                if columns[0] != 'CONTAINER':
                    values = summary.get(columns[0], None)
                    if values is None:
                        values = []

                    values.append(float(columns[2].strip('%')) / 100)
                    summary[columns[0]] = values
            
        # print(summary)
        stats_data[sut_version] = summary
       
    return stats_data

def plot_mean_latencies(log_file_query_paths):
    version_means = extract_mean_latency(log_file_query_paths) 
   
    sut_means = {}
    for sut_version, container_stats in version_means.items():
        first_two_items = list(container_stats.items())[:2]

        first_key, version_1_means = first_two_items[0]
        version_1_means = [value for value in version_1_means]
        count_above_0_6_v1 = sum(1 for value in version_1_means if value > 0.6)

        second_key, version_2_means = first_two_items[1]
        version_2_means = [value for value in version_2_means]
        count_above_0_6_v2 = sum(1 for value in version_2_means if value > 0.6)

        # Generate x values (indices) for the data points
        x_values = np.arange(len(version_1_means))
        x_values2 = np.arange(len(version_2_means))

        # Create a line plot
        plt.figure(figsize=(8, 4))
        plt.plot(x_values, version_1_means, marker='o', linestyle='-', color='b', label='Values')
        plt.plot(x_values, version_2_means, marker='s', linestyle='-', color='r', label='Values')

        # Add labels and title
        plt.xlabel('Data Point Index')
        plt.ylabel('Value')
        plt.title('Line Diagram of Values')
        plt.legend()

        # Save the plot
        plt.savefig(f'stats-{sut_version}.png')
        plt.close()

        # Create a bar plot comparing values above 0.7 vs total
        plt.figure(figsize=(6, 4))
        labels = ['Version 1', 'Version 2']
        above_0_6_counts = [count_above_0_6_v1, count_above_0_6_v2]
        total_counts = [len(version_1_means), len(version_2_means)]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, above_0_6_counts, width, label='Above 0.6')
        rects2 = ax.bar(x + width/2, total_counts, width, label='Total')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Counts')
        ax.set_title('Counts of Values Above 0.6 vs Total')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Save the bar plot
        plt.savefig(f'comparison-{sut_version}.png')
        plt.close()

if __name__ == "__main__":
    # Example log_file_paths, replace with actual paths from queriesPaths.py
    stats_file_paths = getStatsFilePaths()

    result = plot_mean_latencies(stats_file_paths)
    # plot_mean_latencies(log_file_paths)
