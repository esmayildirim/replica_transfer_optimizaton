
# Parallel Data Transfer Experiments with AWS S3 and EC2

## Overview
This system conducts **parallel data transfers** from AWS S3 buckets to EC2 instances and collects **performance metrics** using AWS CloudWatch. The data is used to build multivariate time series for intelligent load balancing and throughput prediction in cloud storage systems.

It consists of:
- `download.py`: Downloads S3 files to EC2, simulating load conditions with Poisson-based delays.
- `threaded.py`: Orchestrates parallel downloads across multiple EC2 clients, collects metrics, and stores them in DynamoDB.

## Requirements
- Python 3.7+
- AWS account with:
  - S3 buckets populated with test datasets (1MB, 100MB, 1GB files)
  - EC2 instances (e.g., `t2.micro`, `t2.xlarge`, `t3.small`)
  - AWS CloudWatch enabled
  - A DynamoDB table named `s3-metrics`
- SSH key pairs for each region (e.g., `xxxx-key-virginia.pem`, `xxxx-key-ohio.pem`)
- Python libraries:
  pip install boto3 numpy simplejson matplotlib python-dateutil
  

## Usage

1. **Prepare your EC2 instances**
   - Ensure `download.py` is present on all EC2 clients
   - Ensure SSH access is configured (e.g., `ssh -i mohab-key.pem ec2-user@<ip>`)

2. **Run from local orchestrator machine**
   ```bash
   python3 threaded.py <FileSizeLabel> <PoissonWait> <bucket1> <bucket2> ...
   ```

   **Example:**
   ```bash
   python3 threaded.py 1MB 2 esma-bucket1 esma-bucket2 esma-bucket3
   ```

   - `1MB.zip`: File size category
   - `2`: Poisson constant (average wait in seconds between downloads)
   - `esma-bucket*`: Buckets to cycle through

## Experiment Scenarios
Each experiment wave uses:
- Different EC2 instance types
- Various file sizes (1MB, 100MB, 1GB)
- Inter-region or intra-region S3/EC2 combinations
- Varying Poisson wait times (0–4 seconds)

## Output & Metrics
- Metrics collected: `BytesDownloaded`, `TotalRequestLatency`
- Granularity: 1-minute intervals
- Stored in DynamoDB under `s3-metrics` table as aligned multivariate time series

