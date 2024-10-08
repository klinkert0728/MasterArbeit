#!/bin/bash
cd "$(dirname "$0")" # Go to the script's directory
export VM_MICRO_IP=$1
export VM_APPLICATION_IP=$2
export VM_APPLICATION_CLIENT_IP=$3

. ../mo
cat ansible/host_template.yml | mo > ../hosts.yml