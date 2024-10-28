#!/usr/bin/env bash

set -euo pipefail

# ssh_user and private key to needed to  copy the basic replica configuration.
keypair_name=$1
keypair_file=$2

# Get ip of victoria-metrics micro.
VM_MICRO_IP=($(gcloud compute instances list --filter="tags.items=vm-micro" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

VM_APPLICATION_IP=($(gcloud compute instances list --filter="tags.items=vm-application" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

VM_APPLICATION_CLIENT_IP=($(gcloud compute instances list --filter="tags.items=vm-application-client" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))

# Pass the ips to teh host file to replace the template file.
./setup/configureHostFile.sh $VM_MICRO_IP $VM_APPLICATION_IP $VM_APPLICATION_CLIENT_IP $VM_APPLICATION_CLIENT_IP

# Copy tsbs script to VM_APPLICATION_CLIENT_IP  .
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/configure_tsbs.sh $keypair_name@$VM_APPLICATION_CLIENT_IP:~/

ansible-playbook -i hosts.yml setup/ansible/configure_gcp_instances.yml --ssh-common-args='-o StrictHostKeyChecking=no'