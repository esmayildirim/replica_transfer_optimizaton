#This is the graphing part of the dataset
import boto3, time, threading, subprocess, sys, datetime
import matplotlib.pyplot as plt
from dateutil.tz import tzutc
from decimal import Decimal
import numpy as np
from os import path
import simplejson as json
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

class DecimalEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Decimal):
      return float(obj)
    return json.JSONEncoder.default(self, obj)



#main


if __name__ == "__main__":
    dataset = pull_dataset("s3-metrics")
    #stringObject = json.dumps(dataset, cls=DecimalEncoder)
    outfile = open("dataset_wAll.json", "w")
    json.dump(dataset, outfile, cls=DecimalEncoder,indent=4)
    outfile.close()
    
