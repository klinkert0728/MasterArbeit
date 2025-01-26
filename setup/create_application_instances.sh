#!/usr/bin/env bash

set -euo pipefail

run=$1
standalone=$2

# define file name as id_rsa and id_rsa.pub
keypair_file="bench_dk_id_rsa"
# define keypair name used as benchUser, be aware that if you change this user, you will need to change the remote_user in all plays for ansible
keypair_name="benchUser"

# GoogleCloud project
PROJECT=master-437610

APPLICATION_BENCHMARK_INSTANCE_NAME=application-sut-experiment-$run

APPLICATION_BENCHMARK_CLIENT_INSTANCE_NAME=application-client-experiment-$run

# define file name as id_rsa and id_rsa.pub
keypair_file="bench_dk_id_rsa"
# define keypair name used as benchUser, be aware that if you change this user, you will need to change the remote_user in all plays for ansible
keypair_name="benchUser"

# set default region and zone
export CLOUDSDK_COMPUTE_REGION=europe-west1
export CLOUDSDK_COMPUTE_ZONE="${CLOUDSDK_COMPUTE_REGION}-b"

echo "starting instances application benchmark..."
#create application benchmark instance sut
gcloud compute instances create $APPLICATION_BENCHMARK_INSTANCE_NAME --project=$PROJECT --image-family=debian-11 --zone=$CLOUDSDK_COMPUTE_ZONE --image-project=debian-cloud  --machine-type=e2-standard-4 --create-disk=auto-delete=yes,size=50 --tags=vm-application-$run,http-server,https-server
gcloud compute instances add-metadata $APPLICATION_BENCHMARK_INSTANCE_NAME --zone=$CLOUDSDK_COMPUTE_ZONE --metadata-from-file ssh-keys="./id_rsa_formatted.pub"

# create application benchmark client
gcloud compute instances create $APPLICATION_BENCHMARK_CLIENT_INSTANCE_NAME --project=$PROJECT --image-family=debian-11 --zone=$CLOUDSDK_COMPUTE_ZONE --image-project=debian-cloud  --machine-type=e2-standard-8 --create-disk=auto-delete=yes,boot=yes,device-name=instance-20241102-174601,image=projects/debian-cloud/global/images/debian-12-bookworm-v20241009,mode=rw,size=50,type=pd-balanced --tags=vm-application-client-$run,http-server,https-server
gcloud compute instances add-metadata $APPLICATION_BENCHMARK_CLIENT_INSTANCE_NAME --zone=$CLOUDSDK_COMPUTE_ZONE --metadata-from-file ssh-keys="./id_rsa_formatted.pub"


if [[ "$standalone" == "true" ]]; then
    # configure environment
    sleep 15
    VM_APPLICATION_IP=($(gcloud compute instances list --filter="tags.items=vm-application-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))
    VM_APPLICATION_CLIENT_IP=($(gcloud compute instances list --filter="tags.items=vm-application-client-$run" --format="value(EXTERNAL_IP)"  | tr '\n' ' '))
    scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/configure_tsbs.sh $keypair_name@$VM_APPLICATION_CLIENT_IP:~/
    scp -i "./$keypair_file" -o StrictHostKeyChecking=no ./setup/scripts/dockerStats.sh $keypair_name@$VM_APPLICATION_IP:~/

    ./setup/configure_application_environment.sh $VM_APPLICATION_IP $VM_APPLICATION_CLIENT_IP $run
    ansible-playbook -i hosts_application_$run.yml setup/ansible/configure_gcp_instances.yml --ssh-common-args='-o StrictHostKeyChecking=no'
    echo "This script was run directly from the terminal."
else
    echo "This script was called from another script or process."
fi
