import boto3
import time
import datetime
import os
import sys
import random
import numpy as np

# Initialize the S3 resource using boto3
s3 = boto3.resource("s3")

def download(s3Client, bucketList, s3Folder, destFolder, poissonConstant, threadid):
    """
    Downloads files from a list of S3 buckets to a local destination folder.
    Parameters:
        s3Client (boto3.resource): The S3 resource object.
        bucketList (list): A list of S3 bucket names to cycle through.
        s3Folder (str): The S3 folder path to download from.
        destFolder (str): The local directory to save downloaded files to.
        poissonConstant (int): Controls the wait time between downloads (simulates bursty or steady traffic).
        threadid (int): A unique ID for the current thread to help balance S3 bucket selection.
    Returns:
        (float, float, str): Tuple of (startTime, endTime, dateString)
    """
    
    #print(bucketList)

    # Deterministic bucket selection based on thread ID
    counter = threadid % len(bucketList)
    bucketName = bucketList[counter]
    #print(counter)
    counter += 1

    # Select the appropriate S3 bucket
    myBucket = s3Client.Bucket(bucketName)

    # Record start time of the session
    startTime = time.time()
    date = time.ctime(startTime)

    # Gather list of files to download and calculate total dataset size
    fileList = []
    totalSize = 0
    for s3Object in myBucket.objects.filter(Prefix=s3Folder):
        #print(s3Object.key, type(s3Object.key))
        fileList.append(s3Object.key)
        totalSize += s3Object.size

    # Begin downloading files from the list
    for i in range(len(fileList)):
        # Skip if it's a folder path (ends with '/')
        if fileList[i][-1] == "/":
            continue

        # Compute local target path for the downloaded file
        target = (
            fileList[i]
            if destFolder is None
            else os.path.join(destFolder, os.path.relpath(fileList[i], s3Folder))
        )

        # Create directories if needed
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        # Re-select a bucket (round-robin if multiple buckets)
        bucketName = bucketList[counter % len(bucketList)]
        counter += 1
        myBucket = s3Client.Bucket(bucketName)

        # Download file from S3 to local destination
        myBucket.download_file(fileList[i], target)

        # Optional wait to simulate network traffic variability
        if poissonConstant != 0:
            waitTime = np.random.poisson(poissonConstant)
            time.sleep(waitTime)

    # Record end time after all files downloaded
    endTime = time.time()

    return (startTime, endTime, date)


# Entry point: retrieves arguments and begins off download process
# Usage: python download.py <s3Folder> <poissonConstant> <threadid> <bucket1> <bucket2> ...
startTime, endTime, date = download(
    s3Client=s3,
    bucketList=sys.argv[4:],          
    s3Folder=sys.argv[1],             
    destFolder="./files",             
    poissonConstant=int(sys.argv[2]), 
    threadid=int(sys.argv[3])         
)

