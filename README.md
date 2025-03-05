# Benchmark Environment Setup Guide

A comprehensive guide for setting up a distributed benchmarking environment for Victoria Metrics on Google Cloud Platform (GCP). This environment enables performance testing and comparison of different Victoria Metrics versions.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Setup Process](#setup-process)
- [Usage Guide](#usage-guide)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Cleanup](#cleanup)

## Overview

Victoria Metrics is a fast, cost-effective, and scalable time series database. This benchmarking environment helps evaluate its performance across different versions and configurations.

## Prerequisites

- Google Cloud SDK (version 400.0.0 or later)
- Ansible (version 2.9 or later)
- Access to a GCP project with the following permissions:
  - `compute.instances.*`
  - `compute.firewalls.*`
  - `iam.serviceAccounts.*`
- Python 3.8 or later
- Bash shell environment

## System Requirements

| Component | Instance Type | vCPUs | Memory | Disk |
|-----------|--------------|--------|---------|------|
| Micro Benchmark | e2-standard-2 | 2 | 8GB | 50GB |
| Application Benchmark | e2-standard-4 | 4 | 16GB | 100GB |
| Application Client | e2-standard-8 | 8 | 32GB | 100GB |
| Controller | e2-medium | 2 | 4GB | 20GB |

## Setup Process

### 1. Create GCP Instances

Run the setup script with an experiment run number:

```bash
./setup_benchmark_environment.sh <experiment_run_number>
```

This script:
- Generates SSH keys for instance access
- Creates four GCP instances with Debian 11
- Configures firewall rules for ports 22, 8428, 8429, 80, and 8080
- Sets up SSH access using the generated keys

### 2. Environment Configuration

After instance creation, the script automatically:
1. Retrieves IP addresses for all instances
2. Configures host files for Ansible
3. Copies necessary scripts to respective instances:
   - TSBS configuration script to Application Client
   - Ansible playbooks to Controller
   - Docker stats script to Application Benchmark instance

### 3. Instance Configuration via Ansible

The setup runs several Ansible playbooks that configure:

#### Common Setup (All Instances)
- Updates system packages
- Installs Git, Curl, and Python pip
- Configures Go environment

#### Docker Setup (Application Benchmark Instance)
- Installs Docker and dependencies
- Configures Docker service
- Sets up Python Docker SDK

#### Victoria Metrics Setup
- Pulls Victoria Metrics Docker images (two versions for comparison)
- Runs Victoria Metrics containers with:
  - 1.5 CPU limit
  - 6GB memory limit
  - Exposed ports 8428 and 8429

#### Controller Setup
- Installs Ansible
- Creates results directory
- Configures cron job for benchmark automation

#### Benchmark Client Setup
- Clones TSBS repository
- Configures benchmark tools
- Sets up connection to Victoria Metrics instances

## Directory Structure

```bash
MasterArbeit/setup/
├── ansible/                    # Ansible playbooks
├── scripts/                    # Configuration scripts
└── create_gcp_instances.sh     # Main setup script
```

## Usage

1. Start the setup:
```bash
./setup/create_gcp_instances.sh <run_number>
```

2. Wait for the setup to complete (approximately 15-20 minutes)

3. Access the controller instance to manage benchmarks:
```bash
ssh -i bench_dk_id_rsa benchUser@<controller-ip>
```

4. Important Note: After configuration completes, both the application benchmark and microbenchmark will start automatically. However, they need to be manually stopped once they finish running. Before stopping the benchmarks:

   a. Retrieve the benchmark results using the appropriate Ansible playbook:
   
   For application benchmark results:
   ```bash
   ansible-playbook retrieve_data_application_benchmark_client.yml -i hosts.yml
   ```
   
   For micro benchmark results:
   ```bash
   ansible-playbook retrieve_data_micro_benchmark_client.yml -i hosts.yml
   ```
   
   For both benchmarks:
   ```bash
   ansible-playbook retrieve_all_data.yml -i hosts.yml
   ```
   
   This will collect all results from the respective benchmark instances into the `./results/<run_number>` directory on the controller.

   b. After confirming the results are collected, you can safely stop the benchmarks.

You can monitor the benchmarks' progress through the logs and results in the `~/results` directory on the controller instance.

## Monitoring & Logging

The environment includes:
- Docker stats monitoring on the Application Benchmark instance
- Victoria Metrics metrics exposure on ports 8428 and 8429
- System resource monitoring via default GCP monitoring
- Benchmark results collection in `~/results` directory


## Cleanup

To delete all created instances:
```bash
gcloud compute instances delete micro-sut-experiment-<run> application-sut-experiment-<run> application-client-experiment-<run> controller-experiment-<run> --zone=europe-west1-b --quiet
```
