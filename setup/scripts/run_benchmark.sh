#! /bin/bash
ansible-playbook -i hosts.yml ./configure_micro_benchmark_client.yml --ssh-common-args='-o StrictHostKeyChecking=no' &
MICRO_PID_1=$!
ansible-playbook -i hosts.yml ./configure_application_benchmark_client.yml --ssh-common-args='-o StrictHostKeyChecking=no' &
APPLICATION_PID_2=$!
wait $MICRO_PID_1 $APPLICATION_PID_2