# Replica Transfer Optimizaton

This project provides several time series data models for multi-step prediction of replica transfer metrics in the AWS cloud. 

Download the repository into your system. 

If you would like to use a virtual environment, create one with the following commands from a terminal application after you install `Python 3.10`  and `virtualenv` library in your system:

```
virtualenv venv --python=python3.10
source ./venv/bin/activate 
```
Install the libraries using the following command:
```
python3.10 -m pip install tensorflow==2.13.0
```
Repeat the command for the other libraries or use requirements.txt file in the software folder you downloaded from github.
```
python3.10 -m pip install -r requirements.txt
```
Once you have everything in place, go inside models/training_testing folder and run each program: 
```
python3.10 mlp_model_mv_mi.py
```
For state-of-the-art models, go inside models/state_of_the_art folder and run sota_models.py:
```
python3.10 sota_models.py
```

The results will include MAE, MAPE and R2 score metrics for BytesDownloaded (BD) and TotalRequestLatency (TRL) metrics for a  many-to-many communication setting in the AWS cloud. 
