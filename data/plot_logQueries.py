import os
import re
import matplotlib.pyplot as plt
import sys
from queriesPaths import getFilePaths
import numpy as np

def extract_mean_latency(file_paths):
    logInsertSummaryResult = []
    logInsertMapWithRunNumber = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read()

        file_name = os.path.basename(file_path)
        runNumber = file_name.split('-')[0]
        logInsertSummary = logInsertMapWithRunNumber.get(runNumber, None)
        
        if logInsertSummary is None:
            logInsertSummary = []
         
        # Use regex to find the mean latency in the summary section
       # Use regex to extract the summary metrics
        mean_rate_match = re.search(r'mean rate (\d+\.\d+) metrics/sec', content)
        mean_rate = float(mean_rate_match.group(1)) if mean_rate_match else None

        query_rate_match = re.search(r'Overall query rate (\d+\.\d+) queries/sec', content)
        query_rate = float(query_rate_match.group(1)) if query_rate_match else None

        latency_match = re.search(r'min:\s+(\d+\.\d+)ms, med:\s+(\d+\.\d+)ms, mean:\s+(\d+\.\d+)ms, max:\s+(\d+\.\d+)ms, stddev:\s+(\d+\.\d+)ms, sum:\s+(\d+\.\d+)sec', content)
        if latency_match:
            min_latency, med_latency, mean_latency, max_latency, stddev_latency, total_latency_time = map(float, latency_match.groups())
        else:
            min_latency = med_latency = mean_latency = max_latency = stddev_latency = total_latency_time = None

        currentLogInsertSummary = {}
        if file_path.endswith('other.log'):
            currentLogInsertSummary['mean_latency'] = mean_latency
            currentLogInsertSummary['version'] = 1
            currentLogInsertSummary['sut_version'] = file_path.split('/')[-3]
            logInsertSummary.append(currentLogInsertSummary)
        else:
            currentLogInsertSummary['mean_latency'] = mean_latency
            currentLogInsertSummary['version'] = 2
            currentLogInsertSummary['sut_version'] = file_path.split('/')[-3]
            logInsertSummary.append(currentLogInsertSummary)
        
        logInsertMapWithRunNumber[runNumber] = logInsertSummary

    return logInsertMapWithRunNumber

def plot_mean_latencies(log_file_query_paths):
    version_means = extract_mean_latency(log_file_query_paths)
    available_sut_versions = set()
    for runNumber, means in version_means.items():
        means = sorted(means, key=lambda x: x['sut_version'])
        available_sut_versions.add(means[0]['sut_version'])
 
        # Plot mean latency for each run with bars for both versions
    # Prepare data for plotting

    sut_means = {}
    sut_versions_colors = set()
    for sut_version in available_sut_versions:
        # Prepare data for plotting
        sut_versions_colors.add(sut_version.split("-")[0])
        sut_versions_colors.add(sut_version.split("-")[1])
        for runNumber, means in version_means.items():
            means = [entry for entry in means if entry['sut_version'] == sut_version]
            logInsertSummary = sut_means.get(sut_version, None)
            if logInsertSummary is None:
                logInsertSummary = []

            if len(means) > 0:
                logInsertSummary.append(means)
                sut_means[sut_version] = logInsertSummary

    color_map = {'v106': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 'v104': (1.0, 0.4980392156862745, 0.054901960784313725, 1.0), 'v107': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0), 'v108': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0), 'v105': (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0), 'v109': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0)}
    for sut_version, means in sut_means.items():
        run_numbers = list(range(len(means)))
        version_1_means = [next((entry['mean_latency'] for entry in means if entry['version'] == 1), 0) for means in means]
        version_2_means = [next((entry['mean_latency'] for entry in means if entry['version'] == 2), 0) for means in means]

        # Create a line plot with markers
        fig, ax = plt.subplots()
        ax.plot(run_numbers, version_1_means, marker='o', linestyle='-', label=f'{sut_version.split("-")[0]}', color=color_map[sut_version.split("-")[0]])
        ax.plot(run_numbers, version_2_means, marker='s', linestyle='-', label=f'{sut_version.split("-")[1]}', color=color_map[sut_version.split("-")[1]])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Mean Query rate (ms)')
        # ax.set_title(f'Mean Latency by Version and Run for {sut_version}')
        ax.set_xticks(run_numbers)
        ax.set_xticklabels(run_numbers)
        ax.legend()

        # Add a grid
        ax.grid(True)

        # Save the plot
        plt.savefig(f'mean_query_rate_all_runs-{sut_version}.png')
        plt.close()

if __name__ == "__main__":
    # Example log_file_paths, replace with actual paths from queriesPaths.py
    csv_file_paths, log_file_paths, query_file_paths = getFilePaths()

    result = plot_mean_latencies(query_file_paths)
    # plot_mean_latencies(log_file_paths)
