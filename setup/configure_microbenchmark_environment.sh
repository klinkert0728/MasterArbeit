#!/bin/bash
cd "$(dirname "$0")" # Go to the script's directory
export VM_MICRO_IP=$1
export VM_CONTROLLER_IP=$2
export RUN=$3

. ../mo
cat ansible/host_micro_template.yml | mo > ../hosts_micro_$RUN.yml