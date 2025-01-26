mkdir -p /home/benchUser/results
while true; do docker stats --no-stream | tee -a /home/benchUser/results/stats.txt; sleep 1; done