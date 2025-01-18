#!/bin/bash
cd "$(dirname "$0")" # Go to the script's directory
export VM_APPLICATION_IP=$1
export VM_APPLICATION_CLIENT_IP=$2
export RUN=$3

. ../mo
cat ansible/host_application_template.yml | mo > ../hosts_application_$RUN.yml