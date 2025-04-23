import datetime
import pandas as pd
import json
from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, TimesNet
from neuralforecast.losses.numpy import mae, mape
from sklearn.metrics import r2_score
#HYPER PARAMETERS
INP = 15
H = 15
STEP_SIZE = 1000

#DATASET LOCATION
DATA_PATH = "../../data/dataset_w7.json"

def get_bytes_downloaded_in_Mbps(data_item, replica_index):
    bd_list = []
    ds = []
    data_points = data_item['Replicas'][replica_index]
    start_time = float(data_item['StartTime'])

    for data_point in data_points['Datapoints']:
        bd_list.append(int(float(data_point['BytesDownloadedSum'])*8/1024/1024/60))
        ds.append(start_time)
        start_time += 60
    return bd_list, ds

def get_trl_in_millisecs(data_item, replica_index):
    trl_list = []
    ds = []
    data_points = data_item['Replicas'][replica_index]
    start_time = float(data_item['StartTime'])

    for data_point in data_points['Datapoints']:
        trl_list.append(int(float(data_point['TotalRequestLatencySum']) /1000))
        ds.append(start_time)
        start_time += 60
        
    return trl_list,ds

def prep_data(metric, horizon_p, input_size_p):
    file = open(DATA_PATH)
    items = json.load(file)
    unique_id= 0
    for index in range(3):
        for item in items:
            if metric == "BD":
                y, ds = get_bytes_downloaded_in_Mbps(item, index)
            else:
                y, ds = get_trl_in_millisecs(item, index)

            for i in range(len(ds)):
                ds[i] = datetime.datetime.fromtimestamp(int(ds[i]))
            if(len(ds) < H + INP):
               continue
            if unique_id == 0:
                df = pd.DataFrame({'ds': ds, 'y': y})
                df['unique_id'] = str(unique_id)
            else:
                df_new =pd.DataFrame({'ds': ds, 'y': y})
                df_new['unique_id'] = str(unique_id)
                df = pd.concat([df, df_new])
            unique_id += 1
    errors_df = sota_model(df, horizon_p, input_size_p )
    print("ERROR METRICS-"+ str(INP)+'_'+str(H)+"-----------------------------------------------")
    print(errors_df)

def sota_model(df, horizon_p, input_size_p):
    df['ds'] = pd.to_datetime(df['ds'])

    print(df.head())

    models = [NHITS(h=horizon_p,
               input_size=input_size_p,
               max_steps=STEP_SIZE),
              NBEATS(h=horizon_p,
               input_size=input_size_p,
               max_steps=STEP_SIZE),
              TimesNet(h=horizon_p,
                 input_size=input_size_p,
                 max_steps=STEP_SIZE)]

    nf = NeuralForecast(models=models, freq='min')
    preds_df = nf.cross_validation(df=df, step_size=1, n_windows=4)
    data = {'N-HiTS': [mae(preds_df['NHITS'], preds_df['y']), mape(preds_df['NHITS'], preds_df['y']), r2_score(preds_df['NHITS'], preds_df['y'])],
        'N-BEATS': [mae(preds_df['NBEATS'], preds_df['y']), mape(preds_df['NBEATS'], preds_df['y']), r2_score(preds_df['NBEATS'], preds_df['y'])],
        'TimesNet': [mae(preds_df['TimesNet'], preds_df['y']), mape(preds_df['TimesNet'], preds_df['y']), r2_score(preds_df['TimesNet'], preds_df['y'])]}
    
    metrics_df = pd.DataFrame(data=data)
    metrics_df.index = ['mae', 'mape', 'r2']

    return metrics_df

if __name__ == "__main__":
    prep_data("BD", H,INP)
    prep_data("TRL", H,INP)
