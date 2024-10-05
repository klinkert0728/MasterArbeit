#!/usr/bin/env bash

set -euo pipefail


run=$1 # determines the experiment run number

# GoogleCloud project
PROJECT=master-437610

# set the primary name
MICRO_BENMARK_INSTANCE_NAME=micro-experiment-$run

APPLICATION_BENMARK_INSTANCE_NAME=application-experiment-$run

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
# create microbenchmark instance
gcloud compute instances create $MICRO_BENMARK_INSTANCE_NAME --project=$PROJECT --image-family=debian-11 --zone=$CLOUDSDK_COMPUTE_ZONE --image-project=debian-cloud  --machine-type=e2-medium --create-disk=auto-delete=yes --tags=vm-micro,http-server,https-server
gcloud compute instances add-metadata $MICRO_BENMARK_INSTANCE_NAME --zone=$CLOUDSDK_COMPUTE_ZONE --metadata-from-file ssh-keys="./id_rsa_formatted.pub"

#create application benchmark instance
gcloud compute instances create $APPLICATION_BENMARK_INSTANCE_NAME --project=$PROJECT --image-family=debian-11 --zone=$CLOUDSDK_COMPUTE_ZONE --image-project=debian-cloud  --machine-type=e2-medium --create-disk=auto-delete=yes --tags=vm-application,http-server,https-server
gcloud compute instances add-metadata $APPLICATION_BENMARK_INSTANCE_NAME --zone=$CLOUDSDK_COMPUTE_ZONE --metadata-from-file ssh-keys="./id_rsa_formatted.pub"


# add firewall rules for SSH, ICMP, victoria-metrics for all VMs
if gcloud compute firewall-rules list --filter="name~allow-victoria-metrics-firewall" | grep -c allow-victoria-metrics-firewall==0; then
    gcloud compute firewall-rules create "allow-victoria-metrics-firewall" --action=ALLOW --rules=icmp,tcp:22,tcp:8428,udp:8428,tcp:80,tcp:8080 --source-ranges=0.0.0.0 --direction=INGRESS
else
    echo "firewall rule already created"
fi

echo "Wait for the instances to spin up"
sleep 15

# configure environment
./setup/configure_environment.sh $keypair_name $keypair_file

# # configure benchmark clients
# ./configure_benchmark_clients.sh