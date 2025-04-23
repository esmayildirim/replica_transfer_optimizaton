import boto3, time, threading, subprocess, sys, datetime
import matplotlib.pyplot as plt
from dateutil.tz import tzutc
from decimal import Decimal
import numpy as np
from os import path
import simplejson as json


# Custom encoder for converting Decimal types (from DynamoDB) into float before storing or logging.

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert Decimal to float
            return float(obj)
        # Use default encoding for all other types
        return json.JSONEncoder.default(self, obj)

# Creates an SSH session to a given EC2 instance and runs 'download.py' remotely.
# Injects experiment parameters (file size, wait time, thread ID, bucket names) through the command line.

def ssh_connection(hostip, cmd, threadid, key):
    # Path to the SSH private key used for login
    Sshkeylocation = key
    user = "ec2-user"

    # Construct the full SSH command, which runs 'download.py' on the remote instance
    ssh_cmd = f"ssh -i {Sshkeylocation} -o ServerAliveInterval=60 -o ServerAliveCountMax=360 {user}@{hostip} {cmd} {sys.argv[1]} {sys.argv[2]} {threadid} {' '.join([str(i) for i in sys.argv[3:]])}"
    print(ssh_cmd)
    p = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for command to complete and capture standard output and error
    out, err = p.communicate()
    print(out.decode(), err.decode())

# Retrieves 'BytesDownloaded' metric from CloudWatch for a specific S3 bucket and time range.

def getBytesDownloadedStat(start, end, region, bucketName):
    # Initialize a CloudWatch client in the correct region
    cw = boto3.client('cloudwatch', region_name=region)

    # Request sum of bytes downloaded, aggregated per minute
    response = cw.get_metric_statistics(
        Namespace='AWS/S3',
        MetricName='BytesDownloaded',
        Dimensions=[
            {'Name': 'BucketName', 'Value': bucketName},
            {'Name': 'FilterId', 'Value': 'BytesDownloaded'}
        ],
        StartTime=start,
        EndTime=end,
        Period=60,  # 1-minute granularity
        Statistics=['Sum'],
        Unit='Bytes'
    )

    return response

# Retrieves 'TotalRequestLatency' metric from CloudWatch for a specific S3 bucket and time range.

def getTotalRequestLatencyStat(start, end, region, bucketName):
    cw = boto3.client('cloudwatch', region_name=region)

    response = cw.get_metric_statistics(
        Namespace='AWS/S3',
        MetricName='TotalRequestLatency',
        Dimensions=[
            {'Name': 'BucketName', 'Value': bucketName},
            {'Name': 'FilterId', 'Value': 'TotalRequestLatency'}
        ],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=['Sum'],
        Unit='Milliseconds'
    )

    return response

# Sorts datapoints returned from CloudWatch into chronological order.

def sortDatapoints(response):
    L = []

    # Create tuples of (original index, timestamp in minutes)
    for i in range(len(response['Datapoints'])):
        timeData = response['Datapoints'][i]['Timestamp']
        sortValue = timeData.hour * 60 + timeData.minute
        L.append((i, sortValue))

    # Use a simple bubble sort to reorder by time
    for i in range(len(L)):
        for j in range(len(L)):
            if L[i][1] < L[j][1]:
                L[i], L[j] = L[j], L[i]

    # Create a new dictionary of sorted datapoints
    response_new = {'Datapoints': [response['Datapoints'][i[0]] for i in L]}

    return response_new

# Aligns timestamps across buckets by padding missing timestamps using adjacent values.

def matchTimestamps(sortedDict):
    alltimestamps = set()

    # Gather all unique timestamps across all buckets
    for datapoints in sortedDict.values():
        for point in datapoints["Datapoints"]:
            alltimestamps.add(point["Timestamp"])

    # For each bucket, find and pad missing timestamps
    for datapoints in sortedDict.values():
        timestamps = set(i["Timestamp"] for i in datapoints["Datapoints"])
        missing = sorted(list(alltimestamps - timestamps))
        largest = max(i["Timestamp"] for i in datapoints["Datapoints"])

        for timestamp in missing:
            # Append at the end if timestamp is newer than the latest
            if timestamp > largest:
                datapoints["Datapoints"].append({
                    "Timestamp": timestamp,
                    "Sum": datapoints["Datapoints"][-1]["Sum"],
                    "Unit": datapoints["Datapoints"][-1]["Unit"]
                })
                continue

            # Insert at correct spot by using the next datapoint's value
            for point in range(len(datapoints["Datapoints"])):
                ctimestamp = datapoints["Datapoints"][point]['Timestamp']
                if timestamp < ctimestamp:
                    datapoints["Datapoints"].insert(point, {
                        "Timestamp": timestamp,
                        "Sum": datapoints["Datapoints"][point+1]["Sum"],
                        "Unit": datapoints["Datapoints"][point+1]["Unit"]
                    })
                    break

    return sortedDict

# Builds a structured JSON object from collected metric data for DynamoDB storage.

def dynamodb_json(startTime, date, bucketdict, avgFileSize, destinationfolder, BDdict, TRLdict, transferType, destinationfolderType, destinationfolderRegion, NumEC2s):
    bucketlst = list(bucketdict.keys())
    sourcelst = list(bucketdict.values())
    replicalst = []
    # For each bucket, attach its region and time-aligned datapoints
    for bucket in range(len(bucketlst)):
        replicalst.append({"SourceRegion": sourcelst[bucket], "Datapoints": []})
        print(bucketlst[0])
        for i in range(len(BDdict[bucketlst[0]]["Datapoints"])):
            timeData = BDdict[bucketlst[0]]["Datapoints"][i]["Timestamp"]

            replicalst[bucket]["Datapoints"].append({
                "Timestamp": [timeData.hour, timeData.minute, timeData.second],
                "BytesDownloadedSum": Decimal(str(BDdict[bucketlst[bucket]]["Datapoints"][i]["Sum"])),
                "TotalRequestLatencySum": Decimal(str(TRLdict[bucketlst[bucket]]["Datapoints"][i]["Sum"]))
            })

    # Complete experiment record to store in DynamoDB
    json_obj = {
        "StartTime": str(startTime),
        "Date": f"{date.split()[0]} {date.split()[1]} {date.split()[2]}",
        "destinationfolderType": destinationfolderType,
        "NumberOfEC2s": NumEC2s,
        "destinationfolderRegion": destinationfolderRegion,
        "TransferType": transferType,
        "destinationfolder": destinationfolder,
        "MeanWaitTimeBetweenRequests": sys.argv[2],
        "AvgFileSize": avgFileSize,
        "Replicas": replicalst,
    }

    return json_obj

# Stores a given JSON record in the specified DynamoDB table.

def put_data(json_item, tableName, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb')

    table = dynamodb.Table(tableName)
    response = table.put_item(Item=json_item)
    return response
'''
# Converts BytesDownloaded values to Mbps per minute.

def get_bytes_downloaded_in_Mbps(data_item):
    bd_list = []

    for data_point in data_item["Datapoints"]:
        bd_list.append(int(float(data_point["BytesDownloadedSum"]) * 8 / 1024 / 1024 / 60))

    return bd_list

# Converts latency (ms) to seconds.

def get_trl_in_millisecs(data_item):
    trl_list = []

    for data_point in data_item["Datapoints"]:
        trl_list.append(int(float(data_point["TotalRequestLatencySum"]) / 1000))

    return trl_list
'''
# Generates a simple list of timestamps (1 through N) for x-axis in graphs.
def get_timestamps_in_minutes(data_item):
    return list(range(1, len(data_item["Datapoints"]) + 1))

# Loads all items from the specified DynamoDB table using paginated scan.

def pull_dataset(table_name, dynamodb=None):
    if not dynamodb:
        dynamodb = boto3.resource("dynamodb")

    table = dynamodb.Table(table_name)
    scan_kwargs = {}
    done = False
    start_key = None
    loop_count = 0

    while not done:
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key

        response = table.scan(**scan_kwargs)
        if loop_count == 0:
            lst = response.get("Items", [])
        else:
            lst += response.get("Items", [])

        loop_count += 1
        start_key = response.get("LastEvaluatedKey", None)
        done = start_key is None

    return lst


# MAIN EXECUTION: Launch remote downloads, collect metrics, align timestamps, upload results.
if __name__ == "__main__":

    # AWS PARAMETERS
    ip_addresses = ['54.175.27.203', '54.159.134.241', '3.87.2.185'] #fill in the public IPs of the EC2 instances
    key_paths = ['/Users/eyildirim/bnlkey.pem'] # fill in the paths of the SSH private key pem files
    # if the EC2 instances are in different regions you will need one pem file per region.
    # fill in the name of the buckets and their regions as a dictionary
    bucketdict = {'esma-bucket1': 'us-east-1', 'esma-bucket2': 'us-east-1', 'esma-bucket3': 'us-east-1'}
    # File size dictionary for each dataset type
    sizedict = {"100MB": 104857600, "1MB": 1048576, "1GB": 1073741824}
    #destination region list
    destRegList = ['us-east-1'] # this is a list if the client VMs are in different regions
    insType = "t2.micro" # instance types of the client VMs
    tType = "intra-region" # inter-region for different region replicas , inter-region-source for multi-region clint VMs
    
    # Define threads for EC2 clients and set up IP addresses and key file locations
    t1 = threading.Thread(target=ssh_connection, args=(ip_addresses[0], "python3 download.py", "0", key_paths[0]))
    t2 = threading.Thread(target=ssh_connection, args=(ip_addresses[1], "python3 download.py", "1", key_paths[0]))
    t3 = threading.Thread(target=ssh_connection, args=(ip_addresses[2], "python3 download.py", "2", key_paths[0]))
    #t4 = threading.Thread(target=ssh_connection, args=(ip_addresses[3], "python3 download.py", "3", key_paths[0]))
    #t5 = threading.Thread(target=ssh_connection, args=(ip_addresses[4], "python3 download.py", "4", key_paths[0]))
    #t6 = threading.Thread(target=ssh_connection, args=(ip_addresses[5], "python3 download.py", "5", key_paths[0]))

    # Start time of experiment
    start = time.time()

    # Launch all transfers in parallel
    t1.start(); t2.start(); t3.start();# t4.start(); t5.start()
    t1.join(); t2.join(); t3.join();# t4.join(); t5.join()

    # Allow time for CloudWatch to aggregate metrics
    end = time.time() + 5 * 60
    #time.sleep(5 * 60)
    time.sleep(5)
    # Gather metrics from CloudWatch for each bucket
    resBD = {}
    resTRL = {}

    for bucket in list(bucketdict.keys()):
        resBD[bucket] = sortDatapoints(getBytesDownloadedStat(start, end, bucketdict[bucket], bucket))
        resTRL[bucket] = sortDatapoints(getTotalRequestLatencyStat(start, end, bucketdict[bucket], bucket))

    # Align timestamps across all buckets
    resBD = matchTimestamps(resBD)
    resTRL = matchTimestamps(resTRL)



    # Create and insert DynamoDB record
    jsonfile = dynamodb_json(
        startTime=start,
        date=time.ctime(start),
        bucketdict=bucketdict,
        avgFileSize=sizedict[sys.argv[1]],
        destinationfolder=f"{sys.argv[1]}p{sys.argv[2]}",
        BDdict=resBD,
        TRLdict=resTRL,
        transferType=tType,
        destinationfolderType=insType,
        destinationfolderRegion=destRegList,
        NumEC2s =  len(ip_addresses)
    )
    put_data(jsonfile, "s3-metrics")

    # Log summary info to text file
    with open("log.txt", "a") as file:
        file.write(f"Date: {datetime.date.today()}|File Size: {sys.argv[1]}|Wait Time: {sys.argv[2]}\n")
