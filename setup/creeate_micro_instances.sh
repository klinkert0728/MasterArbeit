#!/usr/bin/env bash

set -euo pipefail


run=$1 # determines the experiment run number

# GoogleCloud project
PROJECT=master-437610

# set the primary name
MICRO_BENCHMARK_INSTANCE_NAME=micro-sut-experiment-$run


CONTROLLER_INSTANCE_NAME=controller-experiment-$run

# define file name as id_rsa and id_rsa.pub
keypair_file="bench_dk_id_rsa"
# define keypair name used as benchUser, be aware that if you change this user, you will need to change the remote_user in all plays for ansible
keypair_name="benchUser"

# if file does not exist, generate new key pair (id_rsa, id_rsa.pub)
[[ -f "$keypair_file" ]] || {
  echo "generating ssh key..."
  # cc-default as comment and no passphrase in files id_rsa and id_rsa.pub
  ssh-keygen -t rsa -C "$keypair_name" -f "./$keypair_file" -q -N ""
}

# reformat public key to match googles requirements
echo "$keypair_name:$(<./${keypair_file}.pub)" > id_rsa_formatted.pub

# set default region and zone
export CLOUDSDK_COMPUTE_REGION=europe-west1
export CLOUDSDK_COMPUTE_ZONE="${CLOUDSDK_COMPUTE_REGION}-b"

echo "starting instances..."
# create microbenchmark instance sut
gcloud compute instances create $MICRO_BENCHMARK_INSTANCE_NAME --project=$PROJECT --image-family=debian-11 --zone=$CLOUDSDK_COMPUTE_ZONE --image-project=debian-cloud  --machine-type=e2-standard-2 --create-disk=auto-delete=yes,size=50 --tags=vm-micro-$run,http-server,https-server
gcloud compute instances add-metadata $MICRO_BENCHMARK_INSTANCE_NAME --zone=$CLOUDSDK_COMPUTE_ZONE --metadata-from-file ssh-keys="./id_rsa_formatted.pub"

# # create controller
gcloud compute instances create $CONTROLLER_INSTANCE_NAME --project=$PROJECT --image-family=debian-11 --zone=$CLOUDSDK_COMPUTE_ZONE --image-project=debian-cloud  --machine-type=e2-medium --create-disk=auto-delete=yes,size=50 --tags=vm-controller-$run,http-server,https-server
gcloud compute instances add-metadata $CONTROLLER_INSTANCE_NAME --zone=$CLOUDSDK_COMPUTE_ZONE --metadata-from-file ssh-keys="./id_rsa_formatted.pub"

# add firewall rules for SSH, ICMP, victoria-metrics for all VMs
if [ $(gcloud compute firewall-rules list --filter="name~allow-victoria-metrics-firewall" | grep -c allow-victoria-metrics-firewall) -eq 0 ]; then
    gcloud compute firewall-rules create "allow-victoria-metrics-firewall" --action=ALLOW --rules=icmp,tcp:22,tcp:8428,udp:8428,tcp:8429,udp:8429,tcp:80,tcp:8080 --source-ranges=0.0.0.0/0 --direction=INGRESS
else
    echo "Firewall rule already created"
fi

echo "Wait for the instances to spin up"
sleep 15


VM_MICRO_IP=($(gcloud compute instances list --filter="tags.items=vm-micro-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))
VM_CONTROLLER_IP=($(gcloud compute instances list --filter="tags.items=vm-controller-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/configure_tsbs.sh $keypair_name@$VM_MICRO_IP:~/
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/abs/abs_config.json $keypair_name@$VM_MICRO_IP:~/

# Copy ansible scripts to VM_CONTROLLER_IP.
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/ansible/*.yml $keypair_name@$VM_CONTROLLER_IP:~/
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/run_benchmark.sh $keypair_name@$VM_CONTROLLER_IP:~/
scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./bench_dk_id* $keypair_name@$VM_CONTROLLER_IP:~/


./setup/configure_microbenchmark_environment.sh $VM_MICRO_IP $VM_CONTROLLER_IP $run

scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./hosts_micro_$run.yml $keypair_name@$VM_CONTROLLER_IP:~/hosts.yml

ansible-playbook -i hosts_micro_$run.yml setup/ansible/configure_gcp_instances.yml --ssh-common-args='-o StrictHostKeyChecking=no'