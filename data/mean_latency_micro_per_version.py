from queriesPaths import getMicrobenchmarkFilePathsV104vsV105, getMicrobenchmarkFilePathsV105vsV106, getMicrobenchmarkFilePathsV106vsV107, getMicrobenchmarkFilePathsV107vsV108, getMicrobenchmarkFilePathsV108vsV109
import pandas as pd

def load_and_clean_data(csv_file_paths):
    # Load and combine data from multiple CSV files
    columns = ['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op']
    dfs = []
    for file_path in csv_file_paths:
        df = pd.read_csv(file_path, sep=';', names=columns)
        # Replace triple quotes with an empty string in the entire DataFrame
        df = df.applymap(lambda x: x.replace('"""', '').replace('"', '') if isinstance(x, str) else x)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def generate_csv_for_version_comparison(microbenchmark_file_paths, output_file_name):
    df = load_and_clean_data(microbenchmark_file_paths)
    aggregated_data = df.groupby(['test_name', 'version']).agg(
        mean_ms_op=('ms_op', 'mean'),
        median_ms_op=('ms_op', 'median'),
        std_ms_op=('ms_op', 'std')
    ).reset_index()

    pivoted_data = aggregated_data.pivot(index='version', columns='test_name', values='mean_ms_op').reset_index()
    pivoted_data.to_csv(output_file_name, index=False)

def main():
    microbenchmark_file_paths_v104_vs_v105 = getMicrobenchmarkFilePathsV104vsV105()
    microbenchmark_file_paths_v105_vs_v106 = getMicrobenchmarkFilePathsV105vsV106()
    microbenchmark_file_paths_v106_vs_v107 = getMicrobenchmarkFilePathsV106vsV107()
    microbenchmark_file_paths_v107_vs_v108 = getMicrobenchmarkFilePathsV107vsV108()
    microbenchmark_file_paths_v108_vs_v109 = getMicrobenchmarkFilePathsV108vsV109()

    generate_csv_for_version_comparison(microbenchmark_file_paths_v104_vs_v105, 'output_pivoted_v104_vs_v105.csv')
    generate_csv_for_version_comparison(microbenchmark_file_paths_v105_vs_v106, 'output_pivoted_v105_vs_v106.csv')
    generate_csv_for_version_comparison(microbenchmark_file_paths_v106_vs_v107, 'output_pivoted_v106_vs_v107.csv')
    generate_csv_for_version_comparison(microbenchmark_file_paths_v107_vs_v108, 'output_pivoted_v107_vs_v108.csv')
    generate_csv_for_version_comparison(microbenchmark_file_paths_v108_vs_v109, 'output_pivoted_v108_vs_v109.csv')

if __name__ == "__main__":
    main()