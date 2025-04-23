import json
from numpy import array
from numpy import hstack
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import Flatten
from keras.layers import LayerNormalization
from keras.layers import Dropout#
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import callbacks
from sklearn.metrics import r2_score
import statistics as st
from tensorflow.keras.callbacks import ModelCheckpoint
from  tensorflow.keras.callbacks import EarlyStopping

#HYPERPARAMETERS
NUM_FILTERS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 3
NSTEPS_BACK = 15
NSTEPS_FUTURE = 15
EPOCH = 150
#DATA LOCATION
DATA_PATH = '../../data/dataset_w7.json'

def get_bytes_downloaded_in_Mbps(data_item, replica_index):
    bd_list = []
    data_points = data_item['Replicas'][replica_index]
    for data_point in data_points['Datapoints']:
        bd_list.append(int(float(data_point['BytesDownloadedSum'])*8/1024/1024/60))#convert bytes/min to Mbps
    return bd_list

def get_trl_in_millisecs(data_item, replica_index):
    trl_list = []
    data_points = data_item['Replicas'][replica_index]
    for data_point in data_points['Datapoints']:
        trl_list.append(int(float(data_point['TotalRequestLatencySum']) /1000))
    return trl_list

def prepare_dataset_BD(n_steps_back, n_steps_future):
    file = open(DATA_PATH)
    items = json.load(file)
    loop_count = 0
    for item in items:
        bDownload_list_0 = get_bytes_downloaded_in_Mbps(item, 0)
        bDownload_list_1 = get_bytes_downloaded_in_Mbps(item, 1)
        bDownload_list_2 = get_bytes_downloaded_in_Mbps(item, 2)
        
        if loop_count == 0:
            X, y = stack_item_time_series(bDownload_list_0, bDownload_list_1, bDownload_list_2, n_steps_back, n_steps_future)
        else:
            X1, y1 = stack_item_time_series(bDownload_list_0, bDownload_list_1, bDownload_list_2, n_steps_back, n_steps_future)
            X += X1
            y += y1
        loop_count +=1
    return array(X),array(y)

def prepare_dataset_TRL(n_steps_back, n_steps_future):
    file = open(DATA_PATH)
    items = json.load(file)
    loop_count = 0
    for item in items:
        trl_list_0 = get_trl_in_millisecs(item, 0)
        trl_list_1 = get_trl_in_millisecs(item, 1)
        trl_list_2 = get_trl_in_millisecs(item, 2)
        
        if loop_count == 0:
            X, y = stack_item_time_series(trl_list_0, trl_list_1, trl_list_2,n_steps_back, n_steps_future)
        else:
            X1, y1 = stack_item_time_series(trl_list_0, trl_list_1, trl_list_2, n_steps_back, n_steps_future)
            X += X1
            y += y1
        loop_count +=1
    return array(X),array(y)

def stack_item_time_series(series_0, series_1, series_2,  n_steps_in, n_steps_out):
    in_seq1 = array(series_0)
    in_seq2 = array(series_1)
    in_seq3 = array(series_2)
    X,y = split_sequence(in_seq1, n_steps_in, n_steps_out)
    
    X1, y1 = split_sequence(in_seq2, n_steps_in, n_steps_out)
    X2, y2 = split_sequence(in_seq3, n_steps_in, n_steps_out)
    X = np.concatenate((array(X),array(X1)))
    X = np.concatenate((X, array(X2)))
    X = X.tolist()
    
    y = np.concatenate((array(y),array(y1)))
    y = np.concatenate((y, array(y2)))
    y = y.tolist()
    
    return (X,y)

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return X, y

def CNN_model_train(X_train,y_train, metric_name):
    n_features = 1
    n_output = y_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    model = Sequential()
    model.add(Conv1D(NUM_FILTERS,KERNEL_SIZE, activation='relu', input_shape=(NSTEPS_BACK,n_features), kernel_initializer = 'he_normal', padding = 'same'))
    model.add(Conv1D(NUM_FILTERS,KERNEL_SIZE, activation='relu', kernel_initializer = 'he_normal',padding = 'same'))
    model.add(AveragePooling1D(padding = 'same'))
    for i in range(NUM_CONV_LAYERS-1):
        print("Adding additional layers")
        model.add(Conv1D(NUM_FILTERS * 2,KERNEL_SIZE, activation='relu', kernel_initializer = 'he_normal',padding = 'same'))
        model.add(Conv1D(NUM_FILTERS * 2,KERNEL_SIZE, activation='relu', kernel_initializer = 'he_normal',padding = 'same'))
        model.add(AveragePooling1D(padding = 'same'))
    
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_initializer = 'he_normal'))#1st hidden layer
    model.add(Dense(n_output)) # output layer
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae', metrics = ['mae'])# mse is changed to mape
   
    checkpoint = ModelCheckpoint('cnn_best_weights_uv_si_'+metric_name+"_"+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_CONV_LAYERS)+"_"+str(NUM_FILTERS)+'.h5',
                             monitor='val_loss',
                             verbose=0,
                             mode='min',
                             save_best_only=True)
    
    history = model.fit(X_train,
                        y_train,
                        epochs=EPOCH,
                        callbacks = [checkpoint],
                        verbose=1,
                        validation_split=0.2)
    #plt.plot(history.history['mae'])
    #plt.plot(history.history['val_mae'])
    #plt.title('model error')
    #plt.ylabel('error')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')
    #plt.savefig('epocs_cnn_uv_si_'+metric_name+"_"+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_CONV_LAYERS)+"_"+str(NUM_FILTERS)+'.eps', dpi = 300)
    #plt.close()
    return model
    
def CNN_model_test_BD(checkpoint_filepath, X_test, y_test):
    model_BD.load_weights(checkpoint_filepath)
    mape = 0
    mae = 0

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1 ))
    y_hat = model_BD.predict(X_test)
    y_test_reshaped = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    R2 = r2_score(y_test_reshaped, y_hat)
    counter = 0
    for i in range (X_test.shape[0]):
        x_input = X_test[i]
        x_input = x_input.reshape((1, X_test.shape[1] , X_test.shape[2]))
        y_predicted = model_BD.predict(x_input, verbose=0)
        y_test_output = y_test[i]
        y_test_output = y_test_output.reshape((1, y_test.shape[1]))

        for j in range(y_test.shape[1]): # the number of output timestamps
            mae += abs(y_test_output[0][j]-y_predicted[0][j])
            if(y_test_output[0][j] != 0.0):
                mape += abs(y_test_output[0][j]-y_predicted[0][j])/y_test_output[0][j]
            else:
                mape += abs(y_predicted[0][j])
        counter+=1
    mae/= counter*y_test.shape[1]
    mape *= 100/(counter*y_test.shape[1])
    return mae, mape, R2

def CNN_model_test_TRL(checkpoint_filepath, X_test, y_test):
    model_TRL.load_weights(checkpoint_filepath)
    mape = 0
    mae = 0

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1 ))
    y_hat = model_TRL.predict(X_test)
    y_test_reshaped = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    R2 = r2_score(y_test_reshaped, y_hat)
    counter = 0
    for i in range (X_test.shape[0]):
        x_input = X_test[i]
        x_input = x_input.reshape((1, X_test.shape[1] , X_test.shape[2]))
        y_predicted = model_TRL.predict(x_input, verbose=0)
        y_test_output = y_test[i]
        y_test_output = y_test_output.reshape((1, y_test.shape[1]))

        for j in range(y_test.shape[1]): # the number of output timestamps
            mae += abs(y_test_output[0][j]-y_predicted[0][j])
            if(y_test_output[0][j] != 0.0):
                mape += abs(y_test_output[0][j]-y_predicted[0][j])/y_test_output[0][j]
            else:
                mape += abs(y_predicted[0][j])
        
        counter+=1
    mae/= counter*y_test.shape[1]
    mape *= 100/(counter*y_test.shape[1])
    return mae, mape, R2

if __name__ == '__main__':
    Xarray_BD, yarray_BD = prepare_dataset_BD(NSTEPS_BACK, NSTEPS_FUTURE )
    Xarray_TRL, yarray_TRL = prepare_dataset_TRL(NSTEPS_BACK, NSTEPS_FUTURE )
    X_train_BD, X_test_BD, y_train_BD, y_test_BD = train_test_split(Xarray_BD, yarray_BD, random_state = 42, test_size = 0.20)
    X_train_TRL, X_test_TRL, y_train_TRL, y_test_TRL = train_test_split(Xarray_TRL, yarray_TRL, random_state = 42, test_size = 0.20)

    #TRAINING
    model_BD = CNN_model_train(X_train_BD,y_train_BD, "BD")
    model_TRL = CNN_model_train(X_train_TRL,y_train_TRL, "TRL")

    #TESTING
    mae_BD, mape_BD, R2_BD = CNN_model_test_BD('cnn_best_weights_uv_si_BD_'+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_CONV_LAYERS)+"_"+str(NUM_FILTERS)+'.h5', X_test_BD, y_test_BD)
    mae_TRL, mape_TRL, R2_TRL = CNN_model_test_TRL('cnn_best_weights_uv_si_TRL_'+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_CONV_LAYERS)+"_"+str(NUM_FILTERS)+'.h5', X_test_TRL, y_test_TRL)
    print("Metric", "BD", "TRL",  "Average", sep='\t')
    
    print("MAPE ", mape_BD,mape_TRL,(mape_BD+mape_TRL)/2, sep = '\t')
    print("MAE ", mae_BD,mae_TRL,(mae_BD+mae_TRL)/2, sep = '\t')
    print("R2",R2_BD, R2_TRL,  (R2_BD+R2_TRL)/2,sep = '\t')
    

