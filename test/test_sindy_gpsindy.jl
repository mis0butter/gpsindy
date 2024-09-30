using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 
using DataFrames 


## ============================================ ## 

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/25hz_noise_0.1/rollout_1.csv" 

df_best_sindy, df_best_gpsindy, fig_csv = cross_validate_csv( csv_path_file, 1 ) 
fig_csv 

## ============================================ ## 


df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 25, 0.1 ) 


## ============================================ ## 
# sandbox 



