# file: log_parser.py

import re

def parse_queries_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the "all queries" header
    start_index = None
    for i, line in enumerate(lines):
        if line.startswith("Run complete after"):
            start_index = i
            break

    if start_index is None:
        print(file_path)
        raise ValueError("'all queries' header not found in the log file.")

    # Extract the summary statistics line after the "all queries" header
    summary_line = None
    for line in lines[start_index:]:
        if line.startswith("min:"):
            summary_line = line.strip()
            break

    if not summary_line:
        raise ValueError("Summary statistics not found after 'all queries' header.")

    # Extract the mean value using regular expressions
    match = re.search(r"mean:\s+([\d.]+)ms", summary_line)
    if match:
        mean_value = float(match.group(1))
        return { 'mean_latency': mean_value }
    else:
        raise ValueError("Mean value not found in the summary statistics.")