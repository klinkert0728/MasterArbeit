import pandas as pd
import numpy as np
from queriesPaths import getMicrobenchmarkFilePathsV104vsV105, getMicrobenchmarkFilePathsV105vsV106, getMicrobenchmarkFilePathsV106vsV107, getMicrobenchmarkFilePathsV107vsV108, getMicrobenchmarkFilePathsV108vsV109




def transform_data(df):
    new_df = df

    # Pivot the DataFrame
    pivoted_df = new_df.pivot_table(index=['run', 'version'], columns='test_name', values='ms_op', aggfunc='first')

    # Reset the index if you want a clean DataFrame
    pivoted_df.reset_index(inplace=True)


    return pivoted_df


def load_and_clean_data(csv_file_paths):
    # Load and combine data from multiple CSV files
    columns = ['run', 'type', 'test_path', 'test_name', 'version', 'iterations', 'ms_op']
    dfs = []
    for file_path in csv_file_paths:
        # newPath = "../setup/"
        df = pd.read_csv(file_path, sep=';', names=columns)
        # Replace triple quotes with an empty string in the entire DataFrame
        df = df.applymap(lambda x: x.replace('"""', '').replace('"', '') if isinstance(x, str) else x)
        run = df["run"]
        file_name = file_path.split("/")[-1]
        file_name = file_name.split("-")[0]
        df["run"] = file_name
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def aggregate_data(df):
    # Group by 'version' and calculate the mean of 'ms_op'
    mean_df = df.groupby('version').mean().reset_index()
    return mean_df

def log_means_to_file(mean_df, log_file_path):
    # Write the mean data to a log file
    with open(log_file_path, 'w') as log_file:
        log_file.write(mean_df.to_string(index=False))

def main():
    csv_files_v1 = getMicrobenchmarkFilePathsV104vsV105()
    csv_files_v2 = getMicrobenchmarkFilePathsV105vsV106()
    csv_files_v3 = getMicrobenchmarkFilePathsV106vsV107()
    csv_files_v4 = getMicrobenchmarkFilePathsV107vsV108()
    csv_files_v5 = getMicrobenchmarkFilePathsV108vsV109()

    df_v1 = load_and_clean_data(csv_files_v1)
    df_v2 = load_and_clean_data(csv_files_v2)
    df_v3 = load_and_clean_data(csv_files_v3)
    df_v4 = load_and_clean_data(csv_files_v4)
    df_v5 = load_and_clean_data(csv_files_v5)

    df_v1 = transform_data(df_v1)
    df_v2 = transform_data(df_v2)
    df_v3 = transform_data(df_v3)
    df_v4 = transform_data(df_v4)
    df_v5 = transform_data(df_v5)

    df_v1 = df_v1.drop(columns=["run", "-memory.allowedPercent=60"])
    df_v2 = df_v2.drop(columns=["run"])
    df_v3 = df_v3.drop(columns=["run"])
    df_v4 = df_v4.drop(columns=["run"])
    df_v5 = df_v5.drop(columns=["run"])

    # df_v1["version_sut"] = "v104-v105"
    # df_v2["version_sut"] = "v105-v106"
    # df_v3["version_sut"] = "v106-v107"
    # df_v4["version_sut"] = "v107-v108"
    # df_v5["version_sut"] = "v108-v109"

    
    # Aggregate data to get means
    mean_df_v1 = aggregate_data(df_v1)
    mean_df_v2 = aggregate_data(df_v2)
    mean_df_v3 = aggregate_data(df_v3)
    mean_df_v4 = aggregate_data(df_v4)
    mean_df_v5 = aggregate_data(df_v5)

    mean_df_v1['run_time'] = mean_df_v1.sum(axis=1) 
    mean_df_v2['run_time'] = mean_df_v2.sum(axis=1) 
    mean_df_v3['run_time'] = mean_df_v3.sum(axis=1) 
    mean_df_v4['run_time'] = mean_df_v4.sum(axis=1) 
    mean_df_v5['run_time'] = mean_df_v5.sum(axis=1) 

    # Save the means to a CSV file
    mean_df_v1.to_csv('mean_v1.csv', index=False)
    mean_df_v2.to_csv('mean_v2.csv', index=False)
    mean_df_v3.to_csv('mean_v3.csv', index=False)
    mean_df_v4.to_csv('mean_v4.csv', index=False)
    mean_df_v5.to_csv('mean_v5.csv', index=False)



    

if __name__ == "__main__":
    main()