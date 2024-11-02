#!/bin/bash

echo "init tsbs start"

SUT_IP_AND_PORT_LATEST=$1
SUT_IP_AND_PORT_OTHER=$2

# Install required command to generate data from the official tsbs repository.
cd tsbs/cmd/tsbs_generate_data && go install
cd ../tsbs_generate_queries && go install

# go back to home directory
cd

# Install benchmark tool that includes latencies.
go install github.com/loposkin/tsbs@latest

tsbs_dirname=$(ls go/pkg/mod/github.com/loposkin | grep tsbs)

cd go/pkg/mod/github.com/loposkin/$tsbs_dirname/cmd

### install command to load data to sut and run queryes depenedencies
cd tsbs_load_victoriametrics && go install
cd ../tsbs_run_queries_victoriametrics && go install

TSBS_PATH=~/go
export PATH=$PATH:$TSBS_PATH/bin

# # Set the new go path to the environement
echo "PATH=$PATH" | sudo tee -a /etc/environment
echo "TSBS_PATH=$TSBS_PATH" | sudo tee -a /etc/environment

### Generate Data
tsbs_generate_data --use-case=devops --seed=123 --scale=800  --timestamp-start="2024-10-26T00:00:00Z" --timestamp-end="2024-10-31T00:00:00Z" --log-interval="60s" --format="victoriametrics" > ~/data.txt

### Generate Queries
# Keep in mind the timestamp start and end should be the same as the data generated.
tsbs_generate_queries --use-case=devops --seed=123 --scale=800 --timestamp-start="2024-10-26T00:00:00Z" --timestamp-end="2024-10-31T00:00:00Z" --queries=1440 --query-type=double-groupby-all --format=victoriametrics > ~/queries.txt

mkdir -p $HOME/results

### Load data into sut
loadData() {
    IP=$1
    VERSION=$2
    tsbs_load_victoriametrics --urls="http://$IP/write" --workers=10 --batch-size=100 --file="$HOME/data.txt" --latencies-file=$HOME/results/latenciesInserts_${VERSION}.csv 2>&1 | tee $HOME/results/logInserts_${VERSION}.log 
}

loadData $SUT_IP_AND_PORT_LATEST "latest" & 
loadData $SUT_IP_AND_PORT_OTHER "other"

runBenchmark() {
    IP=$1
    VERSION=$2
    tsbs_run_queries_victoriametrics --file=$HOME/queries.txt \
		--urls="http://$IP" \
		--latencies-file=results/latenciesQueries${VERSION}.csv  \
		--print-interval="500" \
		--workers=10 | tee results/logQueries${VERSION}.log

}

runBenchmark $SUT_IP_AND_PORT_LATEST "latest" & 
runBenchmark $SUT_IP_AND_PORT_OTHER "other" &