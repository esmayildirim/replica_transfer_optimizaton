import json
from numpy import array
from numpy import hstack
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
NUM_NODES = 400
NUM_LSTM_LAYERS = 1
NSTEPS_BACK = 15
NSTEPS_FUTURE = 15
EPOCH = 150
#DATASET LOCATION
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

def prepare_dataset(n_steps_back, n_steps_future):
    file = open(DATA_PATH)
    items = json.load(file)
    loop_count = 0
    for item in items:
        bDownload_list_0 = get_bytes_downloaded_in_Mbps(item, 0)
        trl_list_0 = get_trl_in_millisecs(item, 0)
        bDownload_list_1 = get_bytes_downloaded_in_Mbps(item, 1)
        trl_list_1 = get_trl_in_millisecs(item, 1)
        bDownload_list_2 = get_bytes_downloaded_in_Mbps(item, 2)
        trl_list_2 = get_trl_in_millisecs(item, 2)
        
        if loop_count == 0:
            X, y = stack_item_time_series(bDownload_list_0,trl_list_0, bDownload_list_1, trl_list_1, bDownload_list_2, trl_list_2,n_steps_back, n_steps_future)
        else:
            X1, y1 = stack_item_time_series(bDownload_list_0,trl_list_0, bDownload_list_1, trl_list_1, bDownload_list_2, trl_list_2, n_steps_back, n_steps_future)
            X += X1
            y += y1
        loop_count +=1
    return array(X),array(y)

def stack_item_time_series(bDownload_list_0,trl_list_0, bDownload_list_1, trl_list_1, bDownload_list_2, trl_list_2,  n_steps_in, n_steps_out):
    in_seq1 = array(bDownload_list_0)
    in_seq2 = array(trl_list_0)
    in_seq3 = array(bDownload_list_1)
    in_seq4 = array(trl_list_1)
    in_seq5 = array(bDownload_list_2)
    in_seq6 = array(trl_list_2)
    
    in_seq1 = in_seq1.reshape(len(in_seq1), 1)
    in_seq2 = in_seq2.reshape(len(in_seq2), 1)
    in_seq3 = in_seq3.reshape(len(in_seq3), 1)
    in_seq4 = in_seq4.reshape(len(in_seq4), 1)
    in_seq5 = in_seq5.reshape(len(in_seq5), 1)
    in_seq6 = in_seq6.reshape(len(in_seq6), 1)
    
    dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6))
    X,y = split_sequence(dataset, n_steps_in, n_steps_out)
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

def LSTM_model_train(X_train,y_train):
    n_features = X_train.shape[2]
    n_output = y_train.shape[1] * y_train.shape[2]
    y_train = y_train.reshape((y_train.shape[0], n_output))
    model = Sequential()
    if NUM_LSTM_LAYERS == 1:
        model.add(LSTM(NUM_NODES, activation='relu', input_shape = (NSTEPS_BACK, n_features), kernel_initializer = 'he_normal'))
    elif NUM_LSTM_LAYERS == 2:
        model.add(LSTM(NUM_NODES, activation='relu', input_shape = (NSTEPS_BACK, n_features), kernel_initializer = 'he_normal', return_sequences = 'true'))
        model.add(LSTM(NUM_NODES, activation='relu', kernel_initializer = 'he_normal'))
    else:
        model.add(LSTM(NUM_NODES, activation='relu', input_shape = (NSTEPS_BACK, n_features), kernel_initializer = 'he_normal', return_sequences = 'true'))
        for i in range(NUM_LSTM_LAYERS-2):
            model.add(LSTM(NUM_NODES, activation='relu', kernel_initializer = 'he_normal', return_sequences = 'true'))
        model.add(LSTM(NUM_NODES, activation='relu', kernel_initializer = 'he_normal'))
       
    model.add(Dense(400, activation='relu', kernel_initializer = 'he_normal'))
    model.add(Dense(n_output)) # output layer
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae', metrics = ['mae'])
   
    checkpoint = ModelCheckpoint('lstm_best_weights_mv_mi_'+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_LSTM_LAYERS)+"_"+str(NUM_NODES)+'.h5',
                             monitor='val_loss',
                             verbose=1,
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
    #plt.savefig('epocs_lstm'+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_LSTM_LAYERS)+"_"+str(NUM_NODES)+'.eps', dpi = 300)
    #plt.close()
    return model
    
def LSTM_model_test(checkpoint_filepath, X_test, y_test):
    model.load_weights(checkpoint_filepath)
    mape = [0,0,0,0,0,0]
    mae = [0,0,0,0,0,0]
    y_hat = model.predict(X_test)
    y_test_reshaped = y_test.reshape((y_test.shape[0], y_test.shape[1] * y_test.shape[2] ))
    R2 = r2_score(y_test_reshaped, y_hat)
    counter = 0
    for i in range (X_test.shape[0]):
        x_input = X_test[i]
        n_input = X_test.shape[1] * X_test.shape[2]
        n_output = y_test.shape[1] * y_test.shape[2]
        x_input = x_input.reshape((1, X_test.shape[1] , X_test.shape[2]))
        y_predicted = model.predict(x_input, verbose=0)
        y_test_output = y_test[i]
        y_test_output = y_test_output.reshape((1, n_output))
        for j in range(y_test.shape[1]): # the number of output timestamps
            for k in range(6): # number of dimensions
                mae[k] += abs(y_test_output[0][j*6+k]-y_predicted[0][j*6+k])
                if(y_test_output[0][j*6+k] != 0.0):
                    mape[k] += abs(y_test_output[0][j*6+k]-y_predicted[0][j*6+k])/y_test_output[0][j*6+k]
                else:
                    mape[k] += abs(y_predicted[0][j*6+k])
                
        counter+=1
    for j in range(6):
        mae[j] /= counter*y_test.shape[1]
        mape[j] *= 100/(counter*y_test.shape[1])
    return mae, mape, R2

if __name__ == '__main__':
    Xarray, yarray = prepare_dataset(NSTEPS_BACK, NSTEPS_FUTURE)
    X_train, X_test, y_train, y_test = train_test_split(Xarray, yarray, random_state = 42, test_size = 0.20)
    
    #TRAINING
    model = LSTM_model_train(X_train,y_train)
    
    #TESTING
    mae_list, mape_list,  R2 = LSTM_model_test('lstm_best_weights_mv_mi_'+str(NSTEPS_BACK)+"_"+str(NSTEPS_FUTURE)+"_layers"+str(NUM_LSTM_LAYERS)+"_"+str(NUM_NODES)+'.h5', X_test, y_test)
    print("Metric", "Replica1-BD", "Replica1-TRL", "Replica2-BD", "Replica2-TRL", "Replica3-BD",
          "Replica3-TRL", "Average","Avg_BD", "Avg_TRL", sep='\t')
    print("MAPE ", mape_list[0],mape_list[1],mape_list[2],mape_list[3],mape_list[4], mape_list[5],
           st.mean(mape_list), st.mean([mape_list[0],mape_list[2],mape_list[4]]), st.mean([mape_list[1],mape_list[3],mape_list[5]]), sep = '\t' )
    print("MAE ", mae_list[0],mae_list[1],mae_list[2],mae_list[3],mae_list[4], mae_list[5],st.mean(mae_list),
           st.mean([mae_list[0],mae_list[2],mae_list[4]]), st.mean([mae_list[1],mae_list[3],mae_list[5]]), sep = '\t' )
    print("R2_score", R2)
