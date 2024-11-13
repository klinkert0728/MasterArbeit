#!/usr/bin/env bash

set -euo pipefail

# ssh_user and private key to needed to  copy the basic replica configuration.
keypair_name=$1
keypair_file=$2
run=$3

# Get ip of victoria-metrics micro.
VM_MICRO_IP=($(gcloud compute instances list --filter="tags.items=vm-micro-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

VM_APPLICATION_IP=($(gcloud compute instances list --filter="tags.items=vm-application-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

VM_APPLICATION_CLIENT_IP=($(gcloud compute instances list --filter="tags.items=vm-application-client-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

VM_CONTROLLER_IP=($(gcloud compute instances list --filter="tags.items=vm-controller-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

# Pass the ips to teh host file to replace the template file.
./setup/configureHostFile.sh $VM_MICRO_IP $VM_APPLICATION_IP $VM_APPLICATION_CLIENT_IP $run $VM_CONTROLLER_IP

# Copy tsbs script to VM_APPLICATION_CLIENT_IP  .
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/configure_tsbs.sh $keypair_name@$VM_APPLICATION_CLIENT_IP:~/
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/abs/abs_config.json $keypair_name@$VM_MICRO_IP:~/

# Copy ansible scripts to VM_CONTROLLER_IP.
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/ansible/*.yml $keypair_name@$VM_CONTROLLER_IP:~/
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./hosts_$run.yml $keypair_name@$VM_CONTROLLER_IP:~/hosts.yml
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/run_benchmark.sh $keypair_name@$VM_CONTROLLER_IP:~/
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./bench_dk_id* $keypair_name@$VM_CONTROLLER_IP:~/

# Run the ansible playbook to configure the gcp instances.
ansible-playbook -i hosts_$run.yml setup/ansible/configure_gcp_instances.yml --ssh-common-args='-o StrictHostKeyChecking=no'

