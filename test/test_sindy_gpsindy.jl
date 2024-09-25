using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 

## ============================================ ## 


csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/10hz_noise_0.02/rollout_3.csv" 

df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv(csv_path_file) 

f_train 


